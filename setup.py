from setuptools import setup, find_packages

setup(
    name="pallas_tpu_kernel_matmul",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "jax",
        "jaxlib",
        "numpy",
        "tabulate",
        "matplotlib",
    ],
    author="Shanquan Tian",
    author_email="sqsqtian@gmail.com",
    description="TPU MatMul kernels using JAX's Pallas framework",
    keywords="jax, pallas, tpu, matmul, matrix-multiplication, gpu",
    url="https://github.com/sqtian/PALLAS_TPU_KERNEL_MATMUL",
)
