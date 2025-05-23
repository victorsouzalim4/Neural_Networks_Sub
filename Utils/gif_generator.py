import imageio
import os


def generateGif(frame_files, output_filename="perceptron_training.gif", fps=2):
    """
    Gera um GIF a partir de uma lista de arquivos de imagem.

    :param frame_files: Lista de caminhos dos arquivos de imagem (frames).
    :param output_filename: Nome do arquivo GIF de saída.
    :param fps: Frames por segundo do GIF.
    """
    if not os.path.exists("gifs"):
        os.makedirs("gifs")

    output_path = os.path.join("gifs", output_filename)
    images = [imageio.imread(frame) for frame in frame_files]
    imageio.mimsave(output_path, images, fps=fps)

    # Limpeza dos frames temporários
    for frame in frame_files:
        os.remove(frame)

    if os.path.exists("frames"):
        os.rmdir("frames")

    print(f"GIF salvo como {output_filename}")
