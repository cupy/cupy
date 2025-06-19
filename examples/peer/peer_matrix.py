import cupy


def main():
    gpus = cupy.cuda.runtime.getDeviceCount()
    for peerDevice in range(gpus):
        for device in range(gpus):
            if peerDevice == device:
                continue
            flag = cupy.cuda.runtime.deviceCanAccessPeer(device, peerDevice)
            print(
                f'Can access #{peerDevice} memory from #{device}: '
                f'{flag == 1}')


if __name__ == '__main__':
    main()
