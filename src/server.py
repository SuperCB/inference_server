import asyncio
import time
import grpc
from PIL import Image
from io import BytesIO
from inference import inference
from loguru import logger
from inference_pb2_grpc import InferenceServer, add_InferenceServerServicer_to_server
from inference_pb2 import InferenceRequest, InferenceReply


class WorkItem:
    def __init__(self, image: bytes):
        self.image = image


class InferenceService(InferenceServer):

    def __init__(self):
        super(InferenceService, self).__init__()
        self.MAX_BATCH_SIZE = 64
        self.LATENCY = 0.001
        self.item_queue = asyncio.Queue(65)
        task = asyncio.create_task(self.queue_worker())
        self.loop = asyncio.shield(task)

        self.last_run_time = time.perf_counter()

    async def inference_worker(self):
        logger.info("begin inference worker.......")
        image_list = []
        queue = []
        for _ in range(self.item_queue.qsize()):
            image, res_queue = self.item_queue.get_nowait()
            image_list += image
            queue.append(res_queue)
            self.item_queue.task_done()
        if len(image_list):
            image = inference(image_list)
            self.last_run_time = time.perf_counter()
            logger.info(f"result size is {len(image)}")
            for i in range(len(image)):
                await queue[i].put(image[i])

    def open_image(self, image: bytes) -> Image.Image:
        image = Image.open(BytesIO(image))
        return image

    async def queue_worker(self):
        while True:
            logger.info("queue worker start working.......")
            if self.item_queue.qsize() == self.MAX_BATCH_SIZE:
                await self.inference_worker()
            else:
                if (
                    self.item_queue.qsize()
                    and time.perf_counter() - self.last_run_time > self.LATENCY
                ):
                    logger.info(f"size {self.item_queue.qsize()}")
                    await self.inference_worker()
                else:
                    await asyncio.sleep(0.01)

    async def inference(self, request: InferenceRequest, context):
        logger.info(f"[🦾] Received request")
        start = time.perf_counter()
        images = list(map(self.open_image, request.image))
        res_queue = asyncio.Queue(1)
        await self.item_queue.put((images, res_queue))
        preds = await res_queue.get()
        del res_queue
        logger.info(f"[✅] Done in {(time.perf_counter() - start) * 1000:.2f}ms")
        return InferenceReply(pred=[preds])


async def serve():
    server = grpc.aio.server()
    add_InferenceServerServicer_to_server(InferenceService(), server)
    # using ip v6
    adddress = "[::]:50052"
    server.add_insecure_port(adddress)
    logger.info(f"[📡] Starting server on {adddress}")
    await server.start()
    await server.wait_for_termination()


if __name__ == "__main__":
    asyncio.run(serve())
