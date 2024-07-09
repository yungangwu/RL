from krl.learner import Learner, learner_serve
import asyncio

async def run_learner(args, config):
    learner = Learner(config)
    server = await learner_serve(learner, args)
    await learner.update_policy()
    tasks = [
        asyncio.create_task(server.wait_for_termination()),
        asyncio.create_task(learner.learn()),
        asyncio.create_task(learner.sync_policy())
    ]
    await asyncio.gather(*tasks)