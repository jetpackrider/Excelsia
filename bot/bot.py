"""Excelsia on Discord.

She answers everyone who speaks in the channels she is given, and nowhere else.
Each person has their own thread with her; her mood is shared by the channel.
The /reset command clears her memory of the channel and her mood. She keeps
nothing from before it. By default only moderators can use it.
Two settings come from the environment and are never written to a file:

    DISCORD_TOKEN      the bot's token from Discord's developer portal
    EXCELSIA_CHANNEL   the id of her channel (several ids may be separated by commas)

Run it with:  python -m bot.bot
"""
from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path

import discord

from excelsia.lite import LiteGPT

from .talk import NOT_HERE, RESET_REPLY, Room, tidy

log = logging.getLogger("excelsia")
WAITING_LIMIT = 6  # messages allowed to queue; beyond that she lets some go by


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    token = os.environ.get("DISCORD_TOKEN", "").strip()
    channels = {int(c) for c in os.environ.get("EXCELSIA_CHANNEL", "").replace(",", " ").split()}
    if not token or not channels:
        raise SystemExit("Set DISCORD_TOKEN and EXCELSIA_CHANNEL first. See the README.")

    room = Room(LiteGPT(Path(__file__).with_name("excelsia.npz")))
    intents = discord.Intents.none()
    intents.guilds = True
    intents.guild_messages = True
    intents.message_content = True  # needed to read what people say in her channel
    client = discord.Client(
        intents=intents, max_messages=None, chunk_guilds_at_startup=False,
        member_cache_flags=discord.MemberCacheFlags.none(),
        allowed_mentions=discord.AllowedMentions.none())  # she can never ping anyone
    commands = discord.app_commands.CommandTree(client)
    one_at_a_time = asyncio.Lock()
    waiting = 0

    @commands.command(name="reset", description="Clear her memory and her mood in this channel.")
    @discord.app_commands.default_permissions(manage_messages=True)  # moderators, unless the server says otherwise
    @discord.app_commands.guild_only()
    async def reset(interaction: discord.Interaction) -> None:
        if interaction.channel_id not in channels:
            await interaction.response.send_message(NOT_HERE, ephemeral=True)
            return
        room.forget(interaction.channel_id)  # every thread in the channel, and her mood
        await interaction.response.send_message(RESET_REPLY)

    @client.event
    async def on_ready() -> None:
        log.info("she is awake as %s, listening in %d channel(s)", client.user, len(channels))
        for guild in client.guilds:  # register /reset with each server she is in; it shows up at once
            try:
                commands.copy_global_to(guild=guild)
                await commands.sync(guild=guild)
            except discord.HTTPException as problem:
                log.warning("could not set up /reset in %s: %s", guild.name, problem)

    @client.event
    async def on_message(message: discord.Message) -> None:
        nonlocal waiting
        if message.author.bot or message.channel.id not in channels:
            return
        names = (client.user.name, client.user.display_name) if client.user else ()
        text = tidy(message.clean_content, names)
        if not text or room.too_soon(message.author.id) or waiting >= WAITING_LIMIT:
            return
        waiting += 1
        try:
            async with one_at_a_time, message.channel.typing():
                mood, said = await asyncio.to_thread(
                    room.reply, message.channel.id, message.author.id, text)
            if said:
                await message.reply(f"[{mood}] {said}", mention_author=False)
        except discord.HTTPException as problem:
            log.warning("could not answer: %s", problem)
        finally:
            waiting -= 1

    client.run(token, log_handler=None)


if __name__ == "__main__":
    main()
