import numpy as np

# container = None
#
# for i in range(2):
#     temp_container = None
#
#     for j in range(2):
#         k = 2 * i + j
#         m = np.arange(12 * k, 12 * k + 12).reshape(3, 4)
#
#         print(f"m({k}):\n", m)
#
#         if temp_container is None:
#             temp_container = m
#         else:
#             temp_container = np.hstack((temp_container, m))
#
#         print(f"temp_container:\n", temp_container)
#
#     if container is None:
#         container = temp_container
#     else:
#         container = np.vstack((container, temp_container))
#
#     print(f"container:\n", container)

container = None

for i in range(2):
    temp_container = None

    for j in range(2):
        k = 2 * i + j
        m = np.arange(12 * k, 12 * k + 12).reshape((2, 3, 2))

        print(f"m({k}):\n", m)

        if temp_container is None:
            temp_container = m
        else:
            temp_container = np.concatenate((temp_container, m), axis=2)

        print(f"temp_container:\n", temp_container)

    if container is None:
        container = temp_container
    else:
        container = np.concatenate((container, temp_container), axis=1)

    print(f"container:\n", container)

weights = np.random.random((2, 1, 1))
print("weights:\n", weights)

weight_container = weights * container
print("weight_container:\n", weight_container)
