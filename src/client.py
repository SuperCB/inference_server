import asyncio
import threading
from io import BytesIO
from pprint import pformat
import time
import grpc
from loguru import logger
from PIL import Image

from inference_pb2 import InferenceReply, InferenceRequest
from inference_pb2_grpc import InferenceServerStub

image = Image.open("./examples/cat.jpg")
buffered = BytesIO()
image.save(buffered, format="JPEG")
image_bytes = buffered.getvalue()

addr = "[::]:50052"


def parallel_process():
    channel = grpc.insecure_channel(addr)
    stub = InferenceServerStub(channel)

    parallel_num = 10
    samples = 10
    time_counts = [0 for _ in range(parallel_num)]

    def process_loop(idx, num=20):
        tt = 0
        for _ in range(num):
            be = time.time()
            res: InferenceReply = stub.inference(InferenceRequest(image=[image_bytes]))
            af = time.time()
            tt += af - be
        time_counts[idx] = tt

    tasks = []
    for i in range(parallel_num):
        task = threading.Thread(target=process_loop, args=(i, samples))
        task.start()
        tasks.append(task)
    for i in range(parallel_num):
        tasks[i].join()
    qps = (samples * parallel_num) / max(time_counts)
    logger.info(f"qps is {qps}")


async def main():
    async with grpc.aio.insecure_channel(addr) as channel:
        stub = InferenceServerStub(channel)
        start = time.perf_counter()

        res: InferenceReply = await stub.inference(
            InferenceRequest(image=[image_bytes])
        )
        logger.info(
            f"[✅] pred = {pformat(res.pred)} in {(perf_counter() - start) * 1000:.2f}ms"
        )


if __name__ == "__main__":
    parallel_process()
    # asyncio.run(main())
