# import telebot, pika
# from telebot import types
# import os
# import logging

# amqp_url = os.environ["AMQP_URL"]
# connection_parameters = pika.URLParameters(amqp_url)
# keyboard1 = telebot.types.ReplyKeyboardMarkup()
# keyboard1.row('шутка')

# API_TOKEN = '5867678740:AAG1LP0N37Y9v6vYuXd3nfi-rOTrcLSccOM'
# bot = telebot.TeleBot(API_TOKEN)

# @bot.message_handler(commands=['start'])
# def start_message(message):
#     #name = message.from_user.first_name
#     #user_full_name = message.from_user.full_name
#     bot.send_message(message.chat.id, 'Привет, хочешь начать -> нажми на кнопку', reply_markup=keyboard1)

# answer = ''
# @bot.message_handler(content_types=['text'])
# def send_text(message):
#     if message.text =='шутка':
#         bot.send_message(message.chat.id,'Введите начало шутки' )

#         letters = 'йцукенгшщзхъфывапролджэячсмитьбюЙЦУКЕНГШЩЗХЪФЫВАПРОЛДЖЭЯЧСМИТЬБЮ'
#         prompt = 'Доктор'
        
        
#             #connection_parameters = pika.ConnectionParameters('rabbit')

#         connection = pika.BlockingConnection(connection_parameters)
        
#         def on_reply_message_received(ch, method, properties, body):
#             # ch.stop_consuming()
#             body_str = body.decode("utf-8")
#             logging.info(f"Received: {body_str}")
#             global answer
#             answer = body_str
#             logging.info(f"Answer: {answer}")
        
#         channel = connection.channel()
#         channel2 = connection.channel()
#         channel2.queue_declare(queue='reply-queue')
#         channel2.basic_consume(queue='reply-queue',
#                                  on_message_callback=on_reply_message_received)
#         channel.queue_declare(queue='request-queue')
        
#         logging.info(f"Sending Request: {prompt}")
        
#         channel.basic_publish('', routing_key='request-queue', body=prompt)
            
        
#         logging.info("Starting Client")
#         channel2.start_consuming()
    
# if __name__ == '__main__':
#     bot.polling(none_stop=True)

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
    await message.reply('Введите начало шутки')
    req = {'chat_id': message.from_user.id, 'text': 'Доктор'}
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