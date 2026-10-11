"""Keep Windows awake only while a specific training launcher remains alive."""
import argparse
import ctypes


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pid", required=True, type=int)
    args = p.parse_args()
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32]
    kernel.OpenProcess.restype = ctypes.c_void_p
    kernel.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    kernel.CloseHandle.argtypes = [ctypes.c_void_p]
    kernel.SetThreadExecutionState.argtypes = [ctypes.c_uint32]
    kernel.SetThreadExecutionState.restype = ctypes.c_uint32
    handle = kernel.OpenProcess(0x100000, False, args.pid)
    if not handle:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        if not kernel.SetThreadExecutionState(0x80000001):  # continuous + system required; screen can turn off
            raise ctypes.WinError(ctypes.get_last_error())
        print(f"Automatic sleep prevented while training PID {args.pid} is alive", flush=True)
        result = kernel.WaitForSingleObject(handle, 0xFFFFFFFF)
        if result != 0:
            raise ctypes.WinError(ctypes.get_last_error())
    finally:
        kernel.SetThreadExecutionState(0x80000000)
        kernel.CloseHandle(handle)
    print("Training process ended; normal sleep policy restored", flush=True)


if __name__ == "__main__":
    main()
