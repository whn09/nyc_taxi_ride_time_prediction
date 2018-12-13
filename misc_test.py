# coding: utf-8


def test_get_direction_8():
    from load_data import get_direction_8

    test_directions = [-170, 170, 180, -150, -135, -100, -90, -45, -25, -22.5, -22, 0, 22, 22.5, 25, 45, 90, 100, 135, 150]
    true_test_direction_8s = [0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 4, 4, 5, 5, 5, 6, 6, 7, 7]

    test_direction_8s = []
    for test_direction in test_directions:
        test_direction_8s.append(get_direction_8(test_direction))

    print('test_directions:', test_directions)
    print('true_test_direction_8s:', true_test_direction_8s)
    print('test_direction_8s:', test_direction_8s)
    assert(test_direction_8s == true_test_direction_8s)


if __name__ == '__main__':
    test_get_direction_8()
