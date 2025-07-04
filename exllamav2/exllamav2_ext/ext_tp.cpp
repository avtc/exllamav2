#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <unordered_map>

#include "config.h"
#include "ext_tp.h"
#include "cpp/util.h"

#define cuda_check(ans) { gpu_assert((ans), __FILE__, __LINE__); }
inline void gpu_assert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"CUDA error: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

ExtTPContext::ExtTPContext
(
    std::vector<std::tuple<int, int, int>> _kv_split,
    std::vector<std::tuple<int, int, int>> _id_split,
    std::vector<std::tuple<int, int, int>> _vc_split,
    std::vector<std::tuple<int, int, int>> _rs_split,
    std::vector<std::tuple<int, int, int>> _q_split,
    std::vector<torch::Tensor> _pinned_temp,
    std::vector<cudaStream_t> _streams,
    bool _enable_p2p
) :
    kv_split(_kv_split),
    id_split(_id_split),
    vc_split(_vc_split),
    rs_split(_rs_split),
    q_split(_q_split),
    streams(_streams),
    enable_p2p(_enable_p2p)
{
    for (const auto &pt : _pinned_temp)
    {
        void* ptp = (void*) pt.data_ptr();
        pinned_temp.push_back(ptp);
        pinned_size = pt.numel() * pt.element_size();
    }

    // Find max device index
    int max_dev = -1;
    for (int i = 0; i < streams.size(); ++i)
        if (streams[i] && i > max_dev) max_dev = i;

    for (int i = 0; i < streams.size(); ++i) {
        if (streams[i]) {
            all_devices.push_back(i);
            device_streams[i] = streams[i]; // Populate device_streams
        }
    }

    sync_events.resize(max_dev + 1);

    for (int dev : all_devices)
    {
        const at::cuda::OptionalCUDAGuard device_guard(dev); // Use guard
        cuda_check(cudaEventCreateWithFlags(&sync_events[dev], cudaEventDisableTiming));
    }

    #ifdef TP_MULTITHREADED

        int numdevs = all_devices.size();
        thread_pool = new ThreadPool(numdevs);

    #endif

    cudaHostAlloc((void**)&tp_data, sizeof(ExtTPData), cudaHostAllocMapped);
    init_tp_data(tp_data);

    fprintf(stderr, "TP Debug: enable_p2p (initial) = %d\n", (int)enable_p2p);
    can_p2p = false;
    if (enable_p2p) {
        // Check P2P capabilities
        can_p2p = true;
        if (all_devices.size() > 1) {
            for (int i = 0; i < all_devices.size(); ++i) {
                for (int j = i + 1; j < all_devices.size(); ++j) {
                    int canAccess;
                    const at::cuda::OptionalCUDAGuard device_guard(all_devices[i]); // Use guard
                    cuda_check(cudaDeviceCanAccessPeer(&canAccess, all_devices[i], all_devices[j]));
                    if (canAccess == 0) {
                        can_p2p = false;
                        fprintf(stderr, "CUDA Warning: Direct P2P access not available between device %d and %d. Falling back to CPU bounce.\n", all_devices[i], all_devices[j]);
                        break;
                    }
                }
                if (!can_p2p) break;
            }
        } else {
            can_p2p = false; // No P2P needed for single device
        }
    }
    fprintf(stderr, "TP Debug: can_p2p (final) = %d\n", (int)can_p2p);

    if (enable_p2p && can_p2p) {
        fprintf(stderr, "[TP] NCCL: Initializing comms for devices: ");
        for (auto d : all_devices) fprintf(stderr, "%d ", d);
        fprintf(stderr, "\n");
        comms.resize(all_devices.size());
        fprintf(stderr, "[TP] NCCL: Calling ncclCommInitAll for %zu devices.\n", all_devices.size());
        ncclCommInitAll(&comms[0], all_devices.size(), &all_devices[0]);
        fprintf(stderr, "[TP] NCCL: ncclCommInitAll completed.\n");
        comms_index.clear();
        for (int i = 0; i < all_devices.size(); ++i) {
            comms_index[all_devices[i]] = i;
            fprintf(stderr, "[TP] comms_index[%d]=%d comms[%d]=%p\n", all_devices[i], i, i, (void*)comms[i]);
        }
    }

    fprintf(stderr, "[TP] NCCL/P2P: all_devices: ");
    for (auto d : all_devices) fprintf(stderr, "%d ", d);
    fprintf(stderr, "\n");
    for (auto& kv : device_streams) fprintf(stderr, "[TP] device_streams[%d]=%p\n", kv.first, (void*)kv.second);
    fprintf(stderr, "[TP] sync_events size: %zu\n", sync_events.size());
}

ExtTPContext::~ExtTPContext()
{
    #ifdef TP_MULTITHREADED
        delete thread_pool;
    #endif

    if (enable_p2p && can_p2p) {
        for (int i = 0; i < comms.size(); ++i)
            ncclCommDestroy(comms[i]);
    }

    for (int dev : all_devices)
    {
        const at::cuda::OptionalCUDAGuard device_guard(dev);
        cuda_check(cudaEventDestroy(sync_events[dev]));
    }
    cudaFreeHost(tp_data);
}

uintptr_t make_tp_context
(
    std::vector<std::tuple<int, int, int>> kv_split,
    std::vector<std::tuple<int, int, int>> id_split,
    std::vector<std::tuple<int, int, int>> vc_split,
    std::vector<std::tuple<int, int, int>> rs_split,
    std::vector<std::tuple<int, int, int>> q_split,
    std::vector<torch::Tensor> pinned_temp,
    std::vector<uintptr_t> streams,
    bool enable_p2p
)
{
    std::vector<cudaStream_t> streams_;
    for (int i = 0; i < streams.size(); ++i)
        streams_.push_back(reinterpret_cast<cudaStream_t> (streams[i]));

    ExtTPContext* ctx = new ExtTPContext
    (
        kv_split,
        id_split,
        vc_split,
        rs_split,
        q_split,
        pinned_temp,
        streams_,
        enable_p2p
    );

    return reinterpret_cast<uintptr_t> (ctx);
}

void free_tp_context
(
    uintptr_t tp_context
)
{
    ExtTPContext* ctx = reinterpret_cast<ExtTPContext*> (tp_context);
    delete ctx;
}

void tp_broadcast
(
    uintptr_t tp_context,
    int buffer,
    torch::Tensor source,
    int broadcast_type,
    const std::vector<torch::Tensor> &targets,
    int dim,
    int t_device
)
{
    ExtTPContext* ctx = reinterpret_cast<ExtTPContext*> (tp_context);

    size_t size = source.numel() * 2;
    TORCH_CHECK(size <= ctx->pinned_size, "Temporary tensor is too small")

    // If P2P is enabled and available, use NCCL broadcast
    if (ctx->enable_p2p && ctx->can_p2p && ctx->all_devices.size() > 1) {
        int src_dev = source.device().index();
        const at::cuda::OptionalCUDAGuard device_guard(src_dev); // Use guard
        cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

        ncclGroupStart();
        for (int i = 0; i < targets.size(); ++i) {
            int dev = targets[i].device().index();
            int comms_i = ctx->comms_index.at(dev);
            ncclBroadcast(source.data_ptr(), targets[i].data_ptr(), source.numel(), ncclFloat16, src_dev, ctx->comms[comms_i], stream);
        }
        ncclGroupEnd();
    }
    else // Fallback to CPU bounce
    {
        void* source_g = NULL;

        int src_dev = source.device().index();

        if (src_dev >= 0)
        {
            const at::cuda::OptionalCUDAGuard device_guard_src(src_dev); // Use guard
            cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

            source_g = (void*) source.data_ptr();
            cuda_check(cudaMemcpyAsync(ctx->pinned_temp[buffer], source_g, size, cudaMemcpyDeviceToHost, stream));
        }

        std::vector<std::tuple<int, int, int>> split;
        switch(broadcast_type)
        {
            case BROADCAST_KV: split = ctx->kv_split; break;
            case BROADCAST_ID: split = ctx->id_split; break;
            case BROADCAST_VC: split = ctx->vc_split; break;
            case BROADCAST_RS: split = ctx->rs_split; break;
            case BROADCAST_Q: split = ctx->q_split; break;
        }

        for (int i = 0; i < split.size(); ++i)
        {
            int dev = std::get<0>(split[i]);
            if (t_device != -1 && t_device != dev) continue;

            void* target = (void*) targets[i].data_ptr();
            if (target == source_g) continue;

            const at::cuda::OptionalCUDAGuard device_guard_target(dev); // Use guard
            cudaStream_t stream = ctx->device_streams.at(dev);
            cuda_check(cudaMemcpyAsync(target, ctx->pinned_temp[buffer], size, cudaMemcpyHostToDevice, stream));
        }
    }

    tp_cross_device_barrier(tp_context, broadcast_type, t_device);
}

void tp_gather
(
    uintptr_t tp_context,
    int buffer,
    const std::vector<torch::Tensor> &inputs,
    int broadcast_type,
    const std::vector<torch::Tensor> &targets,
    int broadcast_type_target,
    int dim,
    int t_device
)
{
    tp_gather_barrier
    (
        tp_context,
        buffer,
        inputs,
        broadcast_type,
        targets,
        broadcast_type_target,
        dim,
        t_device,
        nullptr
    );
}

void tp_gather_barrier
(
    uintptr_t tp_context,
    int buffer,
    const std::vector<torch::Tensor> &inputs,
    int broadcast_type,
    const std::vector<torch::Tensor> &targets,
    int broadcast_type_target,
    int dim,
    int t_device,
    Barrier* barrier
)
{
    ExtTPContext* ctx = reinterpret_cast<ExtTPContext*> (tp_context);

    std::vector<std::tuple<int, int, int>> split;
    switch(broadcast_type)
    {
        case BROADCAST_KV: split = ctx->kv_split; break;
        case BROADCAST_ID: split = ctx->id_split; break;
        case BROADCAST_VC: split = ctx->vc_split; break;
        case BROADCAST_RS: split = ctx->rs_split; break;
        case BROADCAST_Q: split = ctx->q_split; break;
    }

    int out_rows = inputs[0].size(0);
    int out_cols = std::get<2>(split[split.size() - 1]) * dim;
    int esize = inputs[0].element_size();

    // If P2P is enabled and available, use NCCL AllGather
    if (ctx->enable_p2p && ctx->can_p2p && ctx->all_devices.size() > 1) {
        ncclGroupStart();
        for (int i = 0; i < inputs.size(); ++i) {
            int dev = inputs[i].device().index();
            int comms_i = ctx->comms_index.at(dev);
            ncclAllGather(inputs[i].data_ptr(), targets[i].data_ptr(), inputs[i].numel(), ncclFloat16, ctx->comms[comms_i], ctx->device_streams.at(dev));
        }
        ncclGroupEnd();
    }
    else // Fallback to CPU bounce
    {
        for (int i = 0; i < split.size(); ++i)
        {
            int dev = std::get<0>(split[i]);
            if (t_device != -1 && t_device != dev) continue;

            uint8_t* src = (uint8_t*) inputs[i].data_ptr();
            int src_cols = inputs[i].size(1);
            uint8_t* dst = ((uint8_t*) ctx->pinned_temp[buffer]) + std::get<1>(split[i]) * esize * dim;

            const at::cuda::OptionalCUDAGuard device_guard_src(dev); // Use guard
            cuda_check(cudaMemcpy2DAsync
            (
                dst,
                out_cols * esize,
                src,
                src_cols * esize,
                src_cols * esize,
                out_rows,
                cudaMemcpyDeviceToHost,
                ctx->device_streams.at(dev)
            ));
        }

        if (broadcast_type_target == -2) return;

        if (barrier)
            barrier->arrive_and_wait();

        tp_cross_device_barrier(tp_context, broadcast_type, t_device);

        if (broadcast_type_target == -1) return;

        size_t size = targets[0].numel() * 2;

        switch(broadcast_type_target)
        {
            case BROADCAST_KV: split = ctx->kv_split; break;
            case BROADCAST_ID: split = ctx->id_split; break;
            case BROADCAST_VC: split = ctx->vc_split; break;
            case BROADCAST_RS: split = ctx->rs_split; break;
            case BROADCAST_Q: split = ctx->q_split; break;
        }

        for (int i = 0; i < split.size(); ++i)
        {
            int dev = std::get<0>(split[i]);
            if (t_device != -1 && t_device != dev) continue;

            void* target = (void*) targets[i].data_ptr();

            const at::cuda::OptionalCUDAGuard device_guard_target(dev); // Use guard
            cudaStream_t stream = ctx->device_streams.at(dev);
            cuda_check(cudaMemcpyAsync(target, ctx->pinned_temp[buffer], size, cudaMemcpyHostToDevice, stream));
        }
    }
}

void tp_cross_device_barrier
(
    uintptr_t tp_context,
    int broadcast_type,
    int t_device,
    int stage,
    int next_stage
)
{
    ExtTPContext* ctx = reinterpret_cast<ExtTPContext*> (tp_context);

    std::vector<std::tuple<int, int, int>> split;
    switch(broadcast_type)
    {
        case BROADCAST_KV: split = ctx->kv_split; break;
        case BROADCAST_ID: split = ctx->id_split; break;
        case BROADCAST_VC: split = ctx->vc_split; break;
        case BROADCAST_RS: split = ctx->rs_split; break;
        case BROADCAST_Q: split = ctx->q_split; break;
    }

    if (stage == -1)
    {
        stage = ctx->tp_data->next_stage;
        ctx->tp_data->next_stage = (ctx->tp_data->next_stage + 1) % MAX_SYNC_STAGES;
        next_stage = ctx->tp_data->next_stage;
    }

    uint32_t* sync = ctx->tp_data->sync[stage];
    uint32_t* sync_next = ctx->tp_data->sync[next_stage];

    for (int i = 0; i < ctx->all_devices.size(); ++i)
    {
        int dev = ctx->all_devices[i];
        const at::cuda::OptionalCUDAGuard device_guard(dev); // Use guard
        // if (t_device != -1 && t_device != dev) continue;
        cross_device_barrier_cuda
        (
            ctx->streams[dev],
            sync,
            sync_next,
            ctx->all_devices.size(),
            i
        );
    }

    for (int i = 0; i < ctx->all_devices.size(); ++i)
    {
        int dev = ctx->all_devices[i];
        const at::cuda::OptionalCUDAGuard device_guard(dev); // Use guard
        // if (t_device != -1 && t_device != dev) continue;
        cudaStreamSynchronize(ctx->streams[dev]);
    }

    #ifdef TP_MULTITHREADED
        const at::cuda::OptionalCUDAGuard device_guard(t_device); // Use guard
    #endif

    for (int dev_i : ctx->all_devices) {
        const at::cuda::OptionalCUDAGuard device_guard(dev_i); // Use guard
        cuda_check(cudaEventRecord(ctx->sync_events[dev_i], ctx->device_streams.at(dev_i)));
    }

    for (int dev_j : ctx->all_devices) {
        for (int dev_i : ctx->all_devices) {
            if (dev_i == dev_j) continue;
            const at::cuda::OptionalCUDAGuard device_guard(dev_i); // Use guard
            cuda_check(cudaStreamWaitEvent(ctx->device_streams.at(dev_i), ctx->sync_events[dev_j], 0));
        }
    }
}

//void tp_all_reduce_nccl
//(
//    uintptr_t tp_context,
//    const std::vector<torch::Tensor> &tensors
//)
//{
//    ExtTPContext* ctx = reinterpret_cast<ExtTPContext*> (tp_context);
//
//    ncclGroupStart();
//
//    for (int i = 0; i < tensors.size(); ++i)
//    {
//        int dev = tensors[i].device().index();
//        int comms_i = ctx->comms_index[dev];
//
//        ncclAllReduce
//        (
//            tensors[i].data_ptr(),
//            tensors[i].data_ptr(),
//            tensors[i].numel(),
//            ncclFloat16,
//            ncclSum,
//            ctx->comms[comms_i],
//            ctx->streams[dev]
//        );
//    }
//
//    ncclGroupEnd();
//}

//void tp_all_reduce
//(
//    uintptr_t tp_context,
//    const std::vector<torch::Tensor> &tensors
//)

void tp_all_reduce
(
    uintptr_t tp_context,
    int buffer,
    const std::vector<torch::Tensor> &tensors,
    const std::vector<torch::Tensor> &residuals
)
{
    ExtTPContext* ctx = reinterpret_cast<ExtTPContext*> (tp_context);

    size_t size = tensors[0].numel() * tensors[0].element_size();
    size_t num = tensors.size();

    // If P2P is enabled and available, use NCCL AllReduce
    if (ctx->enable_p2p && ctx->can_p2p && ctx->all_devices.size() > 1) {
        ncclGroupStart();
        for (int i = 0; i < num; ++i) {
            int dev = tensors[i].device().index();
            int comms_i = ctx->comms_index.at(dev);
            ncclAllReduce(tensors[i].data_ptr(), residuals[i].data_ptr(), tensors[i].numel(), ncclFloat16, ncclSum, ctx->comms[comms_i], ctx->device_streams.at(dev));
        }
        ncclGroupEnd();
    }
    else // Fallback to CPU bounce
    {
        for (int i = 0; i < num; ++i)
        {
            int dev = tensors[i].device().index();
            const at::cuda::OptionalCUDAGuard device_guard(dev); // Use guard
            auto torch_stream = at::cuda::getStreamFromExternal(ctx->streams[dev], dev);
            at::cuda::setCurrentCUDAStream(torch_stream);

            if (i > 0)
            {
                int prev_dev = tensors[i - 1].device().index();

                // Copy host buffer to current residual

                cuda_check(cudaStreamWaitEvent
                (
                    ctx->streams[dev],
                    ctx->sync_events[prev_dev],
                    0
                ));

                cuda_check(cudaMemcpyAsync
                (
                    residuals[i].data_ptr(),
                    ctx->pinned_temp[buffer],
                    size,
                    cudaMemcpyHostToDevice,
                    ctx->streams[dev]
                ));
            }

            // Add current tensor to current residual

            residuals[i].add_(tensors[i]);

            // Copy current residual to host buffer

            cuda_check(cudaMemcpyAsync
            (
                ctx->pinned_temp[buffer],
                residuals[i].data_ptr(),
                size,
                cudaMemcpyDeviceToHost,
                ctx->streams[dev]
            ));

            cuda_check(cudaEventRecord
            (
                ctx->sync_events[dev],
                ctx->streams[dev]
            ));
        }

        // Broadcast result

        int last_dev = tensors[num - 1].device().index();

        for (int i = 0; i < num - 1; ++i)
        {
            int dev = tensors[i].device().index();
            const at::cuda::OptionalCUDAGuard device_guard(dev); // Use guard

            cuda_check(cudaStreamWaitEvent
            (
                ctx->streams[dev],
                ctx->sync_events[last_dev],
                0
            ));
            cuda_check(cudaMemcpyAsync
            (
                residuals[i].data_ptr(),
                ctx->pinned_temp[buffer],
                size,
                cudaMemcpyHostToDevice,
                ctx->streams[dev]
            ));
        }
    }
}
