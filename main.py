import discord
import asyncio
import os
from dotenv import load_dotenv
from bot import AImodel

load_dotenv()   

class OSHITBot(discord.Client):
    def __init__(self):
        intents = discord.Intents.default()
        intents.message_content = True        
        
        super().__init__(intents=intents)
        self.ai = AImodel()
        self.cooldown = {}

    async def on_ready(self):
        print(f'Logged in as {self.user}')

    async def on_message(self, message):
        if message.author == self.user:
            return

        if message.author.id in self.cooldown:
            if (discord.utils.utcnow() - self.cooldown[message.author.id]).seconds < 5:
                return
        
        self.cooldown[message.author.id] = discord.utils.utcnow()
        
        async with message.channel.typing():
        
            response = await self.ai.make_response(
                user_id=message.author.id,
                message=message.content
            )
            await message.reply(response[:2000]) 


bot = OSHITBot()
bot.run(os.getenv('DISCORD_TOKEN'))