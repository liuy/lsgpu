import pycuda.driver as cuda

# Dictionary mapping compute capability to cores per SM
cc_cores_per_SM_dict = {
    (2, 0): 32,
    (2, 1): 48,
    (3, 0): 192,
    (3, 5): 192,
    (3, 7): 192,
    (5, 0): 128,
    (5, 2): 128,
    (6, 0): 64,
    (6, 1): 128,
    (7, 0): 64,
    (7, 5): 64,
    (8, 0): 64,
    (8, 6): 128,
    (8, 9): 128,
    (9, 0): 128
}

# Dictionary for Tensor Cores per SM by compute capability
# These are approximate values based on NVIDIA architecture documentation
tensor_cores_per_SM_dict = {
    # Volta (7.0) - 8 Tensor Cores per SM
    (7, 0): 8,
    (7, 5): 8,  # Xavier also has 8 per SM
    # Turing (7.5) - 8 Tensor Cores per SM
    (7, 5): 8,
    # Ampere (8.0, 8.6) - 4 Tensor Cores per SM but they're 2nd gen (more powerful)
    (8, 0): 4,
    (8, 6): 4,
    # Ada Lovelace and Hopper (8.9, 9.0) - 4 Tensor Cores per SM but they're 4th gen
    (8, 9): 4,
    (9, 0): 4
}

# Dictionary for generation names by compute capability
architecture_dict = {
    (2, 0): "Fermi",
    (2, 1): "Fermi",
    (3, 0): "Kepler",
    (3, 5): "Kepler",
    (3, 7): "Kepler",
    (5, 0): "Maxwell",
    (5, 2): "Maxwell",
    (6, 0): "Pascal",
    (6, 1): "Pascal",
    (7, 0): "Volta",
    (7, 5): "Turing",
    (8, 0): "Ampere",
    (8, 6): "Ampere",
    (8, 9): "Ada Lovelace",
    (9, 0): "Hopper"
}

def get_tensor_core_info(compute_capability, sm_count):
    """Get information about Tensor Cores based on compute capability"""
    major, minor = compute_capability

    # Check if architecture supports Tensor Cores
    if major < 7:
        return "Not available (pre-Volta architecture)"

    # Get Tensor Cores per SM
    tensor_cores_per_sm = tensor_cores_per_SM_dict.get(compute_capability)
    if tensor_cores_per_sm is None:
        tensor_cores_per_sm = tensor_cores_per_SM_dict.get((major, 0), "Unknown")

    # Calculate total Tensor Cores
    if isinstance(tensor_cores_per_sm, int):
        total_tensor_cores = tensor_cores_per_sm * sm_count

        # Determine generation
        generation = ""
        if major == 7:
            generation = "1st gen"
        elif major == 8:
            if minor == 6:
                generation = "3nd gen"
            elif minor == 9:
                generation = "4th gen"
            else:
                generation = "2nd gen"

        elif major == 9:
            generation = "4th gen"

        return f"{total_tensor_cores} ({tensor_cores_per_sm}/SM, {generation})"
    else:
        return tensor_cores_per_sm

def print_device_info(device_id):
    try:
        device = cuda.Device(device_id)

        device_name = device.name()

        my_sms = device.get_attribute(cuda.device_attribute.MULTIPROCESSOR_COUNT)
        my_cc = device.compute_capability()
        cores_per_sm = cc_cores_per_SM_dict.get(my_cc, "Unknown")

        if cores_per_sm != "Unknown":
            total_cores = cores_per_sm * my_sms
        else:
            total_cores = "Unknown (compute capability not in database)"

        architecture = architecture_dict.get(my_cc, "Unknown architecture")

        tensor_core_info = get_tensor_core_info(my_cc, my_sms)

        # Get RT Core info (simplified - actual counts vary by specific GPU model)
        rt_cores = "Available" if my_cc >= (7, 5) and my_cc != (8, 0) else "Not available"

        shared_mem = device.get_attribute(cuda.device_attribute.MAX_SHARED_MEMORY_PER_BLOCK)
        shared_mem_kb = shared_mem / 1024

        print(f"\nGPU {device_id}: {device_name}")
        print(f"  Architecture: {architecture}")
        print(f"  Compute capability: {my_cc[0]}.{my_cc[1]}")
        print(f"  Number of SMs: {my_sms}")
        print(f"  CUDA cores per SM: {cores_per_sm}")
        print(f"  Total CUDA cores: {total_cores}")
        print(f"  Tensor Cores: {tensor_core_info}")
        print(f"  RT Cores: {rt_cores}")
        print(f"  Max shared memory per block: {shared_mem_kb} KB")
        try:
            l2_cache_size = device.get_attribute(cuda.device_attribute.L2_CACHE_SIZE)
            l2_cache_mb = l2_cache_size / (1024 * 1024) if l2_cache_size > 0 else 0
            l2_cache_info = f"{l2_cache_mb:.2f} MB" if l2_cache_size > 0 else "Not available"
            print(f"  L2 cache size: {l2_cache_info}")
        except:
            pass

        try:
            clock_rate = device.get_attribute(cuda.device_attribute.CLOCK_RATE)
            print(f"  Clock rate: {clock_rate / 1000:.2f} MHz")
        except:
            pass

        try:
            compute_mode = device.get_attribute(cuda.device_attribute.COMPUTE_MODE)
            modes = {0: "Default", 1: "Exclusive", 2: "Prohibited", 3: "Exclusive Process"}
            mode_str = modes.get(compute_mode, str(compute_mode))
            print(f"  Compute mode: {mode_str}")
        except:
            pass

        return True
    except Exception as e:
        print(f"Error accessing GPU {device_id}: {e}")
        return False

def main():
    # Check if CUDA is available
    try:
        cuda.init()
        # Get count of devices
        device_count = cuda.Device.count()
        print(f"Found {device_count} CUDA-capable device(s)")

        # Print information for each device
        for i in range(device_count):
            print_device_info(i)

    except cuda.RuntimeError as e:
        print(f"CUDA is not available on this system: {e}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
