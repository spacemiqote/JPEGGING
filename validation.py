# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.axes as axes
from scipy.fftpack import dctn, idctn

plt.style.use('dark_background')

# Define a color map
white_cmap = colors.LinearSegmentedColormap.from_list("my_colormap", ["black", "black"])

# Create subplots with a specific size
fig, axs = plt.subplots(6, 3, figsize=(24, 60))

# Define color conversion matrices for RGB to YCbCr and vice versa
RGB_TO_YCBCR = np.array(
    [[0.299, 0.587, 0.114], [-0.169, -0.331, 0.500], [0.500, -0.419, -0.081]]
)
YCBCR_TO_RGB = np.array(
    [[1.000, 0.000, 1.403], [1.000, -0.344, -0.714], [1.000, 1.773, 0.000]]
)


def generate_zigzag_pattern(rows, cols):
    """
    Generates a zigzag pattern of size n x m.

    Parameters:
    - n: The number of rows of the pattern.
    - m: The number of columns of the pattern.
    """
    solution = [[] for _ in range(rows + cols - 1)]

    for i in range(rows):
        for j in range(cols):
            sum = i + j
            if (sum % 2 == 0):
                # add at beginning
                solution[sum].insert(0, (i, j))
            else:
                # add at end of the list
                solution[sum].append((i, j))

    zigzag_pattern = np.zeros((rows, cols), dtype=int)
    counter = 0
    for i in solution:
        for j in i:
            zigzag_pattern[j[0], j[1]] = counter
            counter += 1

    return zigzag_pattern


def zigzag(original_matrix):
    # Generate the zigzag pattern for this matrix size
    zigzag_pattern = generate_zigzag_pattern(*original_matrix.shape)

    # Flatten the input matrix and reorder it according to the zigzag pattern
    return original_matrix.flatten()[np.argsort(zigzag_pattern.flatten())].reshape(original_matrix.shape)


def inverse_zigzag(zigzag_matrix):
    # Generate the zigzag pattern for this matrix size
    zigzag_pattern = generate_zigzag_pattern(*zigzag_matrix.shape)

    # Flatten the input matrix and reorder it according to the inverse zigzag pattern
    return zigzag_matrix.flatten()[zigzag_pattern.flatten().astype(int)].reshape(zigzag_matrix.shape)


def quantization_matrix(quality=50):
    """
    Returns the luminance and chrominance quantized tables for a given quality.

    Parameters:
    - quality: The quality of the quantized tables. Must be between 1 and 100.
    """
    std_lum = np.array(  # standard luminance quantized table
        [[16, 11, 10, 16, 24, 40, 51, 61],
         [12, 12, 14, 19, 26, 58, 60, 55],
         [14, 13, 16, 24, 40, 57, 69, 56],
         [14, 17, 22, 29, 51, 87, 80, 62],
         [18, 22, 37, 56, 68, 109, 103, 77],
         [24, 35, 55, 64, 81, 104, 113, 92],
         [49, 64, 78, 87, 103, 121, 120, 101],
         [72, 92, 95, 98, 112, 100, 103, 99]])

    std_chr = np.array(  # standard chrominance quantized table
        [[17, 18, 24, 47, 99, 99, 99, 99],
         [18, 21, 26, 66, 99, 99, 99, 99],
         [24, 26, 56, 99, 99, 99, 99, 99],
         [47, 66, 99, 99, 99, 99, 99, 99],
         [99, 99, 99, 99, 99, 99, 99, 99],
         [99, 99, 99, 99, 99, 99, 99, 99],
         [99, 99, 99, 99, 99, 99, 99, 99],
         [99, 99, 99, 99, 99, 99, 99, 99]])

    quality_scale = 5000 / quality if (quality < 50) else 200 - quality * 2
    lumin = np.floor((std_lum * quality_scale + 50) / 100).clip(1, 255).astype(int)
    chrom = np.floor((std_chr * quality_scale + 50) / 100).clip(1, 255).astype(int)
    return lumin, chrom


def plot_line(current_ax, y_coordinate, color="#023020", lw=5):
    """
    Plots a horizontal line on the figure.

    Parameters:
    - current_ax: The axis on which to plot the line.
    - y_coordinate: The y coordinate of the line.
    - color: The color of the line.
    - lw: The line width.
    """
    current_ax.plot(
        [0, 1],
        [y_coordinate, y_coordinate],
        color=color,
        lw=lw,
        transform=fig.transFigure,
        clip_on=False,
    )


def convert_color_space(image, conversion_matrix):
    """
    Converts the color space of an image using the provided conversion matrix.

    Parameters:
    - image: The image to be converted.
    - conversion_matrix: The matrix to be used for the conversion.
    """
    return np.dot(image, conversion_matrix.T)


def plot_matrix(current_ax, slice_matrix, matrix_description, round_values):
    """
    Plots a matrix with labels.

    Parameters:
    - current_ax: The axis on which to plot the matrix.
    - slice_matrix: The 2D array to be plotted.
    - matrix_description: The label for the x-axis of the plot.
    - round_values: A boolean indicating whether to round the matrix values.
    """
    # Check that slice_matrix is a 2D array
    assert len(slice_matrix.shape) == 2, "slice_matrix must be a 2D array"
    # Check that current_ax is an AxesSubplot instance (or other suitable type)
    assert isinstance(current_ax, axes.Axes), "current_ax must be an instance of matplotlib.axes.Axes"

    current_ax.imshow(slice_matrix, cmap=white_cmap)
    for (i, j), z in np.ndenumerate(slice_matrix):
        if round_values == 1:  # round values to nearest integer
            current_ax.text(j, i, "{:d}".format(round(z)), ha="center", va="center")
        elif round_values == 0:  # don't round values
            current_ax.text(j, i, "{:d}".format(int(z)), ha="center", va="center")
        elif round_values == 2:  # trim values to 1 decimal place
            current_ax.text(j, i, "{:.1f}".format(z), ha="center", va="center")
    current_ax.set_xticks(np.arange(slice_matrix.shape[1]))
    current_ax.set_yticks(np.arange(slice_matrix.shape[0]))
    current_ax.set_xticks(np.arange(-0.5, slice_matrix.shape[1], 1), minor=True)
    current_ax.set_yticks(np.arange(-0.5, slice_matrix.shape[0], 1), minor=True)
    current_ax.grid(which="minor")
    current_ax.set_xlabel(matrix_description, fontsize=14, fontweight="bold", labelpad=10)
    current_ax.xaxis.tick_top()


# Load the image
img = plt.imread("test16.bmp")

# Convert the image color space from RGB to YCbCr and then back to RGB
ycbcr = convert_color_space(img, RGB_TO_YCBCR)
rgb = convert_color_space(ycbcr, YCBCR_TO_RGB)

# Split the YCbCr matrix into its components and rearranges the blocks
y = ycbcr[:, :, 0].reshape(2, 8, 2, 8).swapaxes(1, 2).reshape(4, 8, 8)
cb = ycbcr[:, :, 1].reshape(2, 8, 2, 8).swapaxes(1, 2).reshape(4, 8, 8)
cr = ycbcr[:, :, 2].reshape(2, 8, 2, 8).swapaxes(1, 2).reshape(4, 8, 8)
y_dct = np.array([dctn(layer - 128, norm='ortho', axes=[0, 1]) for layer in y])
cb_dct = np.array([dctn(layer, norm='ortho', axes=[0, 1]) for layer in cb])
cr_dct = np.array([dctn(layer, norm='ortho', axes=[0, 1]) for layer in cr])
q55_l,q55_c = quantization_matrix(55)
y_dct_q = np.array([(layer / q55_l) for layer in y_dct])
zig_8x8 = generate_zigzag_pattern(8, 8)
np.random.seed(999)
bk=np.random.randint(0,100,[8,8])
print(zigzag(bk))
y_dct_q_zig = np.array([zigzag(layer) for layer in y_dct_q])

# Define the matrices to plot along with their corresponding labels and axes
matrices = [
    (axs[0][0], ycbcr[:, :, 0], "Y Matrix", 0),
    (axs[0][1], ycbcr[:, :, 1], "Cb Matrix", 0),
    (axs[0][2], ycbcr[:, :, 2], "Cr Matrix", 0),
    (axs[1][0], rgb[:, :, 0], "R Matrix", 1),
    (axs[1][1], rgb[:, :, 1], "G Matrix", 1),
    (axs[1][2], rgb[:, :, 2], "B Matrix", 1),
    (axs[2][0], y[0], "Y[0] Matrix", 0),
    (axs[2][1], y[2], "Y[2] Matrix", 0),
    (axs[2][2], cr[3], "Cr[3] Matrix", 0),
    (axs[3][0], y_dct[0], "Y[0] DCT Matrix", 0),
    (axs[3][1], cb_dct[1], "CB[1] DCT Matrix", 0),
    (axs[3][2], cr_dct[3], "CR[3] DCT Matrix", 0),
    (axs[4][0], y_dct[2], "Y[2] DCT Matrix", 2),
    (axs[4][1], q55_l, "Q55 Matrix", 0),
    (axs[4][2], y_dct_q[2], "Y[2] DCT Q Matrix", 1),
    (axs[5][0], zig_8x8, "Zigzag Order Matrix", 0),
    (axs[5][1], y_dct_q[2], "Y[2] DCT Q Matrix", 1),
    (axs[5][2], y_dct_q_zig[2], "Y[2] DCT Q ZigZag Matrix", 1)
]

# Plot the matrices
for ax, matrix, matrix_label, rounding in matrices:
    plot_matrix(ax, matrix, matrix_label, rounding)

# Adjust the space between the subplots
fig.subplots_adjust(hspace=0.2, wspace=0.1, left=0.1)

# Plot the lines
plot_line(axs[0][0], 0.36)
plot_line(axs[1][0], 0.633)

# Display the plot
plt.show()
