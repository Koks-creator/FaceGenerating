from dataclasses import dataclass
from typing import Union, List, Tuple
from pathlib import Path
from tensorflow.keras.models import load_model
from keras import layers
import tensorflow as tf
import numpy as np
from logging import getLogger
import matplotlib.pyplot as plt

from custom_decorators import timeit, log_call
from custom_logger import CustomLogger
from config import Config


class SelfAttention(tf.keras.layers.Layer):
    def __init__(self, channels, **kwargs):
        super().__init__(**kwargs)
        self.channels = channels
        self.q = layers.Conv2D(channels // 8, 1, padding='same')
        self.k = layers.Conv2D(channels // 8, 1, padding='same')
        self.v = layers.Conv2D(channels, 1, padding='same')
        self.gamma = tf.Variable(0.0, trainable=True, dtype=tf.float32)

    def call(self, x):
        b = tf.shape(x)[0]
        h = tf.shape(x)[1]
        w = tf.shape(x)[2]
        c = x.shape[3]

        q = tf.reshape(self.q(x), [b, h*w, c//8])
        k = tf.reshape(self.k(x), [b, h*w, c//8])
        v = tf.reshape(self.v(x), [b, h*w, c])

        attn = tf.nn.softmax(tf.matmul(q, k, transpose_b=True))
        out  = tf.matmul(attn, v)
        out  = tf.reshape(out, [b, h, w, c])
        return self.gamma * out + x

    def get_config(self):
        config = super().get_config()
        config.update({"channels": self.channels})
        return config

logger = getLogger("face_geneator")

if not logger.handlers:
    logger = CustomLogger(logger_log_level=Config.CLI_LOG_LEVEL,
                        file_handler_log_level=Config.FILE_LOG_LEVEL,
                        log_file_name=fr"{Config.ROOT_PATH}/logs/face_generator.log",
                        logger_name="face_geneator"
                        ).create_logger()

@dataclass
class FaceGenerator:
    model_path: Union[str, Path]
    load_attention: bool = False
    latent_dim: int = 100

    @timeit(logger=logger)
    def __post_init__(self) -> None:
        logger.info("Initializing FaceGenerator with: \n" \
        f" - {self.model_path}")
        try:
            if self.load_attention:
                self.generator_model = load_model(self.model_path, custom_objects={
                    "SelfAttention": SelfAttention
                })
            else:
                self.generator_model = load_model(self.model_path)
        except Exception as e:
            logger.error(f"Could not load model from path: {self.model_path}, {e}", exc_info=True)
    
    @log_call(logger=logger, hide_res=True, log_params=["num"])
    @timeit(logger=logger)
    def generate_faces(self, num: int = 1) -> List[np.ndarray]:
        noise = tf.random.normal([num, self.latent_dim]) # x rows of 100 random floats
        generated = self.generator_model(noise, training=False)
        generated = (generated + 1.0) / 2.0  # [-1,1] → [0,1] - because tahn

        return [gen.numpy() for gen in generated]
    
    @staticmethod
    def show_results(results: List[np.ndarray],
                     nrows: int = 0,
                     ncols: int = 0,
                     figsize: Tuple[int, int] = (6, 6)
                    ) -> None:
        res_len = len(results)
        if not nrows or not ncols:
            if res_len < 4:
                ncols = 2
                nrows = res_len // 2
            else:
                ncols = 4
                nrows = res_len // 4

        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
        for i, ax in enumerate(axes.flat):
            ax.imshow(results[i])
            ax.axis('off')
        plt.tight_layout()
        plt.show()



if __name__ == "__main__":
    face_generator = FaceGenerator(
        model_path=Path(Config.MODELS_PATH) / "wganv4_generator.h5",
        load_attention=True,
        latent_dim=100
    )
    res = face_generator.generate_faces(
        num=32
    )
    face_generator.show_results(results=res)
    # img = res[0]
    # img = cv2.resize(img, (256, 256))
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # cv2.imshow("res", img)
    # cv2.waitKey(0)
