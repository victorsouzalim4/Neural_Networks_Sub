import imageio
import os


def generateGif(frame_files, output_filename="perceptron_training.gif", fps=2):
    """
    Generates a GIF from a list of image files.

    :param frame_files: List of image file paths (frames).
    :param output_filename: Name of the output GIF file.
    :param fps: Frames per second of the GIF (controls the speed).
    """

    if not os.path.exists("gifs"):
        os.makedirs("gifs")

    output_path = os.path.join("gifs", output_filename)
    images = [imageio.imread(frame) for frame in frame_files]
    imageio.mimsave(output_path, images, fps=fps)

    # temp frames removed
    for frame in frame_files:
        os.remove(frame)

    if os.path.exists("frames"):
        os.rmdir("frames")

    print(f"GIF salvo como {output_filename}")
