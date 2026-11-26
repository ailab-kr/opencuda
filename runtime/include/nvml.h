/*
 * nvml.h -- declarations of a few NVIDIA Management Library (NVML) names, so
 * programs that load the library at run time (dlopen / GetProcAddress) and
 * only need its types and prototypes compile. Written for this project from
 * NVML's public API reference; no NVML implementation is provided, and the
 * names are only valid with the vendor's library at run time.
 */
#ifndef OCU_NVML_H
#define OCU_NVML_H
#ifdef __cplusplus
extern "C" {
#endif
typedef enum nvmlReturn_enum { NVML_SUCCESS = 0, NVML_ERROR_UNINITIALIZED = 1, NVML_ERROR_INVALID_ARGUMENT = 2,
                               NVML_ERROR_NOT_SUPPORTED = 3, NVML_ERROR_UNKNOWN = 999 } nvmlReturn_t;
typedef struct nvmlDevice_st *nvmlDevice_t;
typedef enum nvmlBrandType_enum {
    NVML_BRAND_UNKNOWN = 0, NVML_BRAND_QUADRO = 1, NVML_BRAND_TESLA = 2, NVML_BRAND_NVS = 3, NVML_BRAND_GRID = 4,
    NVML_BRAND_GEFORCE = 5, NVML_BRAND_TITAN = 6, NVML_BRAND_NVIDIA_VAPPS = 7, NVML_BRAND_NVIDIA_VPC = 8,
    NVML_BRAND_NVIDIA_VCS = 9, NVML_BRAND_NVIDIA_VWS = 10, NVML_BRAND_NVIDIA_CLOUD_GAMING = 11,
    NVML_BRAND_QUADRO_RTX = 12, NVML_BRAND_NVIDIA_RTX = 13, NVML_BRAND_NVIDIA = 14, NVML_BRAND_GEFORCE_RTX = 15,
    NVML_BRAND_TITAN_RTX = 16, NVML_BRAND_COUNT
} nvmlBrandType_t;
nvmlReturn_t nvmlInit_v2(void);
nvmlReturn_t nvmlShutdown(void);
nvmlReturn_t nvmlDeviceGetHandleByPciBusId_v2(const char *pciBusId, nvmlDevice_t *device);
nvmlReturn_t nvmlDeviceGetBrand(nvmlDevice_t device, nvmlBrandType_t *type);
#ifdef __cplusplus
}
#endif
#endif
