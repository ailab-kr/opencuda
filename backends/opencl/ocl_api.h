/*
 * ocl_api.h -- the part of the OpenCL API this backend calls, declared for
 * this project from the Khronos OpenCL 3.0 API specification (types in
 * section 4-5, constants in the tables of each query). The library is
 * found at run time (ocl_load.c), so building needs no OpenCL SDK.
 */
#ifndef OCU_OCL_API_H
#define OCU_OCL_API_H

#include <stddef.h>
#include <stdint.h>

typedef int32_t  cl_int;
typedef uint32_t cl_uint;
typedef uint64_t cl_ulong;
typedef cl_uint  cl_bool;
typedef cl_ulong cl_bitfield;
typedef cl_bitfield cl_device_type;
typedef cl_bitfield cl_mem_flags;
typedef cl_bitfield cl_svm_mem_flags;
typedef cl_bitfield cl_device_svm_capabilities;
typedef cl_bitfield cl_command_queue_properties;
typedef cl_bitfield cl_queue_properties;
typedef intptr_t cl_context_properties;
typedef cl_uint  cl_platform_info;
typedef cl_uint  cl_device_info;
typedef cl_uint  cl_program_build_info;
typedef cl_uint  cl_kernel_work_group_info;
typedef cl_uint  cl_kernel_exec_info;
typedef cl_uint  cl_kernel_info;
typedef cl_uint  cl_event_info;
typedef cl_uint  cl_profiling_info;

typedef struct _cl_platform_id   *cl_platform_id;
typedef struct _cl_device_id     *cl_device_id;
typedef struct _cl_context       *cl_context;
typedef struct _cl_command_queue *cl_command_queue;
typedef struct _cl_mem           *cl_mem;
typedef struct _cl_program       *cl_program;
typedef struct _cl_kernel        *cl_kernel;
typedef struct _cl_event         *cl_event;

#define CL_SUCCESS                         0
#define CL_DEVICE_NOT_FOUND               (-1)
#define CL_MEM_OBJECT_ALLOCATION_FAILURE  (-4)
#define CL_OUT_OF_RESOURCES               (-5)
#define CL_OUT_OF_HOST_MEMORY             (-6)
#define CL_BUILD_PROGRAM_FAILURE          (-11)
#define CL_INVALID_VALUE                  (-30)
#define CL_INVALID_KERNEL_NAME            (-46)
#define CL_INVALID_WORK_GROUP_SIZE        (-54)
#define CL_PLATFORM_NOT_FOUND_KHR         (-1001)

#define CL_TRUE  1u
#define CL_FALSE 0u

/* clGetPlatformInfo */
#define CL_PLATFORM_VERSION 0x0901
#define CL_PLATFORM_NAME    0x0902

/* clGetDeviceIDs / clGetDeviceInfo */
#define CL_DEVICE_TYPE_ALL                0xFFFFFFFFu
#define CL_DEVICE_TYPE                    0x1000
#define CL_DEVICE_MAX_COMPUTE_UNITS       0x1002
#define CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS 0x1003
#define CL_DEVICE_MAX_WORK_GROUP_SIZE     0x1004
#define CL_DEVICE_MAX_WORK_ITEM_SIZES     0x1005
#define CL_DEVICE_ADDRESS_BITS            0x100D
#define CL_DEVICE_MAX_MEM_ALLOC_SIZE      0x1010
#define CL_DEVICE_GLOBAL_MEM_SIZE         0x101F
#define CL_DEVICE_LOCAL_MEM_SIZE          0x1023
#define CL_DEVICE_NAME                    0x102B
#define CL_DEVICE_VENDOR                  0x102C
#define CL_DRIVER_VERSION                 0x102D
#define CL_DEVICE_VERSION                 0x102F
#define CL_DEVICE_EXTENSIONS              0x1030
#define CL_DEVICE_DOUBLE_FP_CONFIG        0x1032
#define CL_DEVICE_HOST_UNIFIED_MEMORY     0x1035
#define CL_DEVICE_OPENCL_C_VERSION        0x103D
#define CL_DEVICE_SVM_CAPABILITIES        0x1053
#define CL_DEVICE_IL_VERSION              0x105B
#define CL_DEVICE_GENERIC_ADDRESS_SPACE_SUPPORT 0x1069
/* cl_intel_device_attribute_query */
#define CL_DEVICE_NUM_SLICES_INTEL        0x4252
#define CL_DEVICE_NUM_SUB_SLICES_PER_SLICE_INTEL 0x4253

#define CL_DEVICE_SVM_COARSE_GRAIN_BUFFER (1u << 0)
#define CL_DEVICE_SVM_FINE_GRAIN_BUFFER   (1u << 1)
#define CL_DEVICE_SVM_FINE_GRAIN_SYSTEM   (1u << 2)
#define CL_DEVICE_SVM_ATOMICS             (1u << 3)

/* contexts and queues */
#define CL_CONTEXT_PLATFORM        0x1084
#define CL_QUEUE_PROPERTIES        0x1093
#define CL_QUEUE_PROFILING_ENABLE  (1u << 1)

/* memory */
#define CL_MEM_READ_WRITE    (1u << 0)
#define CL_MEM_READ_ONLY     (1u << 2)
#define CL_MEM_COPY_HOST_PTR (1u << 5)
#define CL_MEM_SVM_FINE_GRAIN_BUFFER (1u << 10)

/* programs and kernels */
#define CL_PROGRAM_BUILD_LOG           0x1183
#define CL_PROGRAM_BUILD_GLOBAL_VARIABLE_TOTAL_SIZE 0x1185
#define CL_KERNEL_NUM_ARGS             0x1191
#define CL_KERNEL_WORK_GROUP_SIZE      0x11B0
#define CL_KERNEL_LOCAL_MEM_SIZE       0x11B2
#define CL_KERNEL_SPILL_MEM_SIZE_INTEL 0x4109 /* cl_intel_required_subgroup_size */
#define CL_KERNEL_EXEC_INFO_SVM_PTRS   0x11B6

/* events */
#define CL_EVENT_COMMAND_EXECUTION_STATUS 0x11D3
#define CL_COMPLETE  0x0
#define CL_PROFILING_COMMAND_END 0x1283
#define CL_PROFILING_COMMAND_START 0x1282

#define OCL_CALL /* default calling convention on the supported hosts */

typedef struct ocl_api {
    cl_int (*GetPlatformIDs)(cl_uint, cl_platform_id *, cl_uint *);
    cl_int (*GetPlatformInfo)(cl_platform_id, cl_platform_info, size_t, void *, size_t *);
    cl_int (*GetDeviceIDs)(cl_platform_id, cl_device_type, cl_uint, cl_device_id *, cl_uint *);
    cl_int (*GetDeviceInfo)(cl_device_id, cl_device_info, size_t, void *, size_t *);
    cl_context (*CreateContext)(const cl_context_properties *, cl_uint, const cl_device_id *,
                                void (*)(const char *, const void *, size_t, void *), void *,
                                cl_int *);
    cl_int (*ReleaseContext)(cl_context);
    cl_command_queue (*CreateCommandQueueWithProperties)(cl_context, cl_device_id,
                                                         const cl_queue_properties *, cl_int *);
    cl_int (*ReleaseCommandQueue)(cl_command_queue);
    cl_int (*Flush)(cl_command_queue);
    cl_int (*Finish)(cl_command_queue);
    void *(*SVMAlloc)(cl_context, cl_svm_mem_flags, size_t, cl_uint);
    void (*SVMFree)(cl_context, void *);
    cl_int (*EnqueueSVMMemcpy)(cl_command_queue, cl_bool, void *, const void *, size_t, cl_uint,
                               const cl_event *, cl_event *);
    cl_int (*EnqueueSVMMemFill)(cl_command_queue, void *, const void *, size_t, size_t, cl_uint,
                                const cl_event *, cl_event *);
    cl_mem (*CreateBuffer)(cl_context, cl_mem_flags, size_t, void *, cl_int *);
    cl_int (*ReleaseMemObject)(cl_mem);
    cl_program (*CreateProgramWithSource)(cl_context, cl_uint, const char **, const size_t *,
                                          cl_int *);
    cl_int (*BuildProgram)(cl_program, cl_uint, const cl_device_id *, const char *,
                           void (*)(cl_program, void *), void *);
    cl_int (*GetProgramBuildInfo)(cl_program, cl_device_id, cl_program_build_info, size_t, void *,
                                  size_t *);
    cl_int (*ReleaseProgram)(cl_program);
    cl_kernel (*CreateKernel)(cl_program, const char *, cl_int *);
    cl_int (*ReleaseKernel)(cl_kernel);
    cl_int (*SetKernelArg)(cl_kernel, cl_uint, size_t, const void *);
    cl_int (*GetKernelInfo)(cl_kernel, cl_kernel_info, size_t, void *, size_t *);
    cl_int (*SetKernelArgSVMPointer)(cl_kernel, cl_uint, const void *);
    cl_int (*SetKernelExecInfo)(cl_kernel, cl_kernel_exec_info, size_t, const void *);
    cl_int (*GetKernelWorkGroupInfo)(cl_kernel, cl_device_id, cl_kernel_work_group_info, size_t,
                                     void *, size_t *);
    cl_int (*EnqueueNDRangeKernel)(cl_command_queue, cl_kernel, cl_uint, const size_t *,
                                   const size_t *, const size_t *, cl_uint, const cl_event *,
                                   cl_event *);
    cl_int (*EnqueueMarkerWithWaitList)(cl_command_queue, cl_uint, const cl_event *, cl_event *);
    cl_int (*WaitForEvents)(cl_uint, const cl_event *);
    cl_int (*GetEventInfo)(cl_event, cl_event_info, size_t, void *, size_t *);
    cl_int (*GetEventProfilingInfo)(cl_event, cl_profiling_info, size_t, void *, size_t *);
    cl_int (*RetainEvent)(cl_event);
    cl_int (*ReleaseEvent)(cl_event);
    /* OpenCL 2.1; NULL when the library does not have it */
    cl_program (*CreateProgramWithIL)(cl_context, const void *, size_t, cl_int *);
} ocl_api;

/* The OpenCL library's entry points, or NULL when there is none (or it is
   missing an entry point of OpenCL 2.0). Loaded once per process. */
const ocl_api *ocl_load(void);

#endif /* OCU_OCL_API_H */
