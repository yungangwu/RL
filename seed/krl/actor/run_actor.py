from krl.actor import Actor, actor_serve
import asyncio

async def run_actor(args, config):
    actor = Actor(config)
    server = await actor_serve(actor, args.port, args.read_buffer_size, args.write_buffer_size)
    await actor.sync_policy()
    tasks = [
        asyncio.create_task(server.wait_for_termination()),
        asyncio.create_task(actor.send_samples()),
    ]
    await asyncio.gather(*tasks)