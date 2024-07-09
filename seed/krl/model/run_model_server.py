from krl.model import ModelSaver, model_serve
import asyncio

async def run_mode_server(args, config):
    model_saver = ModelSaver(config)
    server = await model_serve(model_saver, args)
    tasks = [
        asyncio.create_task(server.wait_for_termination()),
        asyncio.create_task(model_saver.startup()),
    ]
    await asyncio.gather(*tasks)