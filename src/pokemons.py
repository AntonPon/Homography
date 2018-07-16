import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

image_cropped_path = '../images/pokemons/pokemon_cropped.jpg'
image_full_path = '../images/pokemons/pokemon_full.jpg'


def u2H(u, u0):
    """

    :param u: 2x4 matrix of the coordinates from the first image
    :param u0: 2x4 matrix of the coordinates from the second image
    :return homography_matrix: 3x3 matrix that describes the relationship between coordinates
            of the same ixels at the two images
    """
    matrix_row = 0
    matrix_M = np.zeros((8, 9))
    for i in range(0, u.shape[1]):
        u_1 = u[0, i]
        v_1 = u[1, i]
        u_11 = u0[0, i]
        v_11 = u0[1, i]
        vec_u = np.array([[u_1, v_1, 1, 0, 0, 0,  -u_1*u_11, -v_1*u_11, -u_11]])
        vec_v = np.array([[0, 0, 0, u_1, v_1, 1, -u_1*v_11, -v_1*v_11, -v_11]])
        matrix_M[matrix_row, :] = vec_u

        matrix_M[matrix_row+4, :] = vec_v
        matrix_row += 1
    _, _, d = np.linalg.svd(matrix_M)
    
    homography_matrix = d[-1, :].reshape((3, 3))
    return homography_matrix


def get_pixel(i, j, H, left_im, right_im):
    get_coord = np.dot(H, [j, i, 1])
    get_coord = (get_coord[:2]/get_coord[-1]).astype('int')

    if get_coord[0] >= left_im.shape[0] or get_coord[1] >= left_im.shape[1]:
        return right_im[i, j]
    else:
        return left_im[get_coord[1], get_coord[0]]


def fill_image(left_im, right_im, right_im_color, H):
    result = np.copy(right_im_color)
    for i in range(right_im.shape[0]):
        for j in range(right_im.shape[-1]):
            if is_black(right_im[i, j]):
                result[i, j] = get_pixel(i, j, H, left_im, right_im)
    return result


def is_black(rgb):
    return rgb < 50


if __name__ == '__main__':

    v = np.array([[0, 0, 1, 1], [0, 1, 1, 0]])
    v0 = np.array([[1, 2, 1.5, 1], [1, 2, 0.5, 0]])

    print(u2H(v, v0))

    u = np.array([[946.660550014525, 766.782560278881, 476.049530357316, 338.003631257868,
                   277.347099835383, 361.011281107776, 616.187033988573, 712.400842451825, 846.263532487654, 722.858865110874],
                  [381.415948484555, 364.683112230077, 341.675462380169, 433.706061779801, 550.83591556115,
                   644.958119492592, 613.584051515445, 734.897114360414, 567.568751815629, 421.156434588942]])
    u0 = np.array([[48.6940544204513, 183.602546722184, 456.556938123366, 1049.52682289145, 1538.96228333495, 1730.34409799555,
                    1372.67972305607, 1617.39745327782, 933.442771376005, 503.618040089087],
                   [64.1292243633193, 603.763193570253, 1482.23709693038, 1604.59596204125, 1466.5500629418,
                    1046.13755204803, 522.19061682967, 139.426987508473, 79.8162583518929,628.862447951971]])
    res = u2H(u[:, 4:8], u0[:, 4:8])
    image_full = np.array(Image.open(image_full_path))
    image_croped_grey_scale = np.array(Image.open(image_cropped_path).convert('L'))
    image_croped =  np.array(Image.open(image_cropped_path))
    plt.subplot(121)
    plt.imshow(image_full, cmap='gray')
    plt.plot(u0[0, 4:8], u0[1, 4:8], 'ro')
    plt.subplot(122)
    plt.imshow(image_croped, cmap='gray')
    plt.plot(u[0, 4:8], u[1, 4:8], 'ro')
    plt.savefig('../images/results/cropped_full_pokemon.png')
    plt.close()
    output = fill_image(image_full, image_croped_grey_scale, image_croped, res)
    plt.imshow(output, cmap='gray')
    plt.savefig('../images/results/final_pokemon.png')



