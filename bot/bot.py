import os
import json

import pika
import aio_pika
import asyncio
import logging

from aiogram import Bot, types
from aiogram.dispatcher import Dispatcher
from aiogram.utils import executor
from aiogram.contrib.fsm_storage.memory import MemoryStorage
from aiogram.contrib.middlewares.logging import LoggingMiddleware
from state import BotStates

def json_deserializer(m):
    return json.loads(m.decode('utf-8'))


def json_serializer(data):
    return json.dumps(data).encode("utf-8")



TOKEN = '5867678740:AAG1LP0N37Y9v6vYuXd3nfi-rOTrcLSccOM'
bot = Bot(token=TOKEN)
dp = Dispatcher(bot, storage=MemoryStorage())
dp.middleware.setup(LoggingMiddleware())
amqp_url = os.environ["AMQP_URL"]

conn_params = pika.URLParameters(amqp_url)

async def consume():
    
    connection2 = None
    while connection2 == None:
        try:
            connection2 = await aio_pika.connect_robust(amqp_url)
            logging.info('Connected')
        except:
            logging.info('Waiting for connection')
            await asyncio.sleep(50)

    queue_name = "reply-queue"
    async with connection2:
        channel2 = await connection2.channel()
        await channel2.set_qos(prefetch_count=10)

        queue = await channel2.declare_queue(queue_name)

        async with queue.iterator() as it:
            async for message in it:
                async with message.process():
                    input = json_deserializer(message.body)
                    logging.info(input)
                    await bot.send_message(input['chat_id'], input['answer'])
                    await dp.current_state(user=input["chat_id"]).finish()

@dp.message_handler(state='*', commands=['start'])
async def process_start_command(message: types.Message):
    await dp.current_state(user=message.from_user.id).set_state('bot_wait_a_joke')
    await message.reply('Введите начало шутки')

@dp.message_handler(state=BotStates.BOT_WAIT_A_JOKE)
async def joke_text(message: types.Message):
    await message.reply("Шутка в очереди")
    req = {'chat_id': message.from_user.id, 'text': message.text}
    req = json_serializer(req)
    logging.info(req)
    connection = pika.BlockingConnection(conn_params)
    channel = connection.channel()
    channel.queue_declare(queue='request-queue')
    channel.basic_publish(exchange='', routing_key='request-queue', body=req)
    connection.close()

if __name__ == '__main__':
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.create_task(consume())
    executor.start_polling(dp)