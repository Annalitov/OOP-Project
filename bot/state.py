from aiogram.utils.helper import Helper, HelperMode, ListItem

class BotStates(Helper):
    mode = HelperMode.snake_case

    BOT_WAIT_A_JOKE = ListItem()
    