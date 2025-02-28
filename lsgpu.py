#!/usr/bin/env python3
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

# Dictionary for RT Cores per SM by compute capability
# Based on NVIDIA architecture documentation
rt_cores_per_SM_dict = {
    # Turing (7.5) - 1 RT Core per SM
    (7, 5): 1,
    # Ampere (8.6) - 1 RT Core per SM (2nd gen)
    (8, 6): 1,
    # Ada Lovelace (8.9) - 1 RT Core per SM (3rd gen)
    (8, 9): 1,
    # No RT cores on GA100 (8.0)
    # Hopper (9.0) has no dedicated RT cores
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

# Dictionary mapping compute capability to memory type and clock multiplier
memory_type_dict = {
    (2, 0): ("GDDR5", 2),
    (2, 1): ("GDDR5", 2),
    (3, 0): ("GDDR5", 2),
    (3, 5): ("GDDR5", 2),
    (3, 7): ("GDDR5", 2),
    (5, 0): ("GDDR5", 2),
    (5, 2): ("GDDR5", 2),
    (6, 0): ("HBM2", 1),
    (6, 1): ("GDDR5X", 2),
    (7, 0): ("HBM2", 1),
    (7, 5): ("GDDR6", 2),
    (8, 0): ("HBM2", 1),
    (8, 6): ("GDDR6X", 2),
    (8, 9): ("GDDR6X", 2),
    (9, 0): ("HBM3", 1),
}

def get_tensor_core_info(compute_capability, sm_count):
    """Get information about Tensor Cores based on compute capability"""
    major, minor = compute_capability

    # Check if architecture supports Tensor Cores
    if major < 7:
        return "Not available"

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

def get_rt_core_info(compute_capability, sm_count):
    """Get information about RT Cores based on compute capability"""
    major, minor = compute_capability

    if (major, minor) not in rt_cores_per_SM_dict:
        return "Not available"

    rt_cores_per_sm = rt_cores_per_SM_dict.get(compute_capability)

    total_rt_cores = rt_cores_per_sm * sm_count

    generation = ""
    if major == 7:
        generation = "1st gen"
    elif major == 8:
        if minor == 6:
            generation = "2nd gen"
        elif minor == 9:
            generation = "3rd gen"

    return f"{total_rt_cores} ({rt_cores_per_sm}/SM, {generation})"

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

        rt_core_info = get_rt_core_info(my_cc, my_sms)

        shared_mem = device.get_attribute(cuda.device_attribute.MAX_SHARED_MEMORY_PER_BLOCK)
        shared_mem_kb = shared_mem / 1024

        print(f"\nGPU {device_id}: {device_name}")
        print(f"  Architecture: {architecture}")
        print(f"  Compute Capability: {my_cc[0]}.{my_cc[1]}")
        print(f"  Number of SMs: {my_sms}")
        print(f"  CUDA Cores per SM: {cores_per_sm}")
        print(f"  Total CUDA Cores: {total_cores}")
        print(f"  Tensor Cores: {tensor_core_info}")
        print(f"  RT Cores: {rt_core_info}")
        print(f"  Max Shared Memory per Block: {shared_mem_kb} KB")
        try:
            l2_cache_size = device.get_attribute(cuda.device_attribute.L2_CACHE_SIZE)
            l2_cache_mb = l2_cache_size / (1024 * 1024) if l2_cache_size > 0 else 0
            l2_cache_info = f"{l2_cache_mb:.2f} MB" if l2_cache_size > 0 else "Not available"
            print(f"  L2 Cache Size: {l2_cache_info}")
        except:
            pass

        try:
            clock_rate = device.get_attribute(cuda.device_attribute.CLOCK_RATE)
            print(f"  Clock Rate: {clock_rate / 1000:.2f} MHz")
        except:
            pass

        try:
            memory_clock_rate = device.get_attribute(cuda.device_attribute.MEMORY_CLOCK_RATE)
            memory_clock_mhz = memory_clock_rate / 1000
            memory_info = memory_type_dict.get(my_cc, ("Unknown", 1))
            memory_type = memory_info[0]
            clock_multiplier = memory_info[1]
            memory_bus_width = device.get_attribute(cuda.device_attribute.GLOBAL_MEMORY_BUS_WIDTH)
            memory_size = device.total_memory() / (1024 * 1024 * 1024)  # Convert to GB

            print(f"  Memory Type: {memory_type}")
            print(f"  Memory Size: {memory_size:.2f} GB")
            print(f"  Memory Bus Width: {memory_bus_width}-bit")
            print(f"  Effective Memory Clock Rate: {memory_clock_mhz:.2f} MHz")

            memory_speed = (memory_bus_width / 8) * (memory_clock_mhz / 1000) * clock_multiplier
            print(f"  Memory Speed: {memory_speed:.1f} GB/s")
        except:
            pass

        try:
            compute_mode = device.get_attribute(cuda.device_attribute.COMPUTE_MODE)
            modes = {0: "Default", 1: "Exclusive", 2: "Prohibited", 3: "Exclusive Process"}
            mode_str = modes.get(compute_mode, str(compute_mode))
            print(f"  Compute Mode: {mode_str}")
        except:
            pass

        return True
    except Exception as e:
        print(f"Error accessing GPU {device_id}: {e}")
        return False

def main():
    try:
        cuda.init()
        device_count = cuda.Device.count()
        print(f"Found {device_count} CUDA-capable device(s)")

        for i in range(device_count):
            print_device_info(i)

    except cuda.RuntimeError as e:
        print(f"CUDA is not available on this system: {e}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
