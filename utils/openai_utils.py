import asyncio
import logging
import os
from typing import Any
from aiohttp import ClientSession
from tqdm.asyncio import tqdm_asyncio
import random
from time import sleep

import aiolimiter

import openai
from openai import AsyncOpenAI, OpenAIError

async def _throttled_openai_chat_completion_acreate_parse(
    client: AsyncOpenAI,
    model: str,
    messages,
    temperature: float,
    max_tokens: int,
    top_p: float,
    limiter: aiolimiter.AsyncLimiter,
    response_format,
    n: int = 1,
):
    async with limiter:
        for _ in range(10):
            try:
                return await client.beta.chat.completions.parse(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    response_format=response_format,
                )
            except openai.RateLimitError as e:
                print("Rate limit exceeded, retrying...")
                sleep(random.randint(10, 20))  # 增加重试等待时间
            except openai.BadRequestError as e:
                print(e)
                return None
            except OpenAIError as e:
                print(e)
                sleep(random.randint(5, 10))
            except Exception as e:
                print(e)
        return None

async def generate_from_openai_chat_completion_parse(
    client,
    messages,
    engine_name: str,
    response_format,
    temperature: float = 1.0,
    max_tokens: int = 512,
    top_p: float = 1.0,
    requests_per_minute: int = 100,
):
    # https://chat.openai.com/share/09154613-5f66-4c74-828b-7bd9384c2168
    delay = 60.0 / requests_per_minute
    limiter = aiolimiter.AsyncLimiter(1, delay)
    async_responses = [
        _throttled_openai_chat_completion_acreate_parse(
            client,
            model=engine_name,
            messages=message,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            limiter=limiter,
            response_format=response_format,
        )
        for message in messages
    ]
    
    responses = await tqdm_asyncio.gather(*async_responses)
    
    outputs = []
    for response in responses:
        try:
            outputs.append(response.choices[0].message)
        except:
            outputs.append(None)
    return outputs
    
async def _throttled_openai_chat_completion_acreate(
    client: AsyncOpenAI,
    model: str,
    messages,
    temperature: float,
    max_tokens: int,
    top_p: float,
    limiter: aiolimiter.AsyncLimiter,
):
    async with limiter:
        for _ in range(10):
            try:
                return await client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                )
            except openai.BadRequestError as e:
                print(e)
                return None
            except OpenAIError as e:
                print(e)
                sleep(random.randint(5, 10))
        return None


async def generate_from_openai_chat_completion(
    client,
    messages,
    engine_name: str,
    temperature: float = 1.0,
    max_tokens: int = 512,
    top_p: float = 1.0,
    requests_per_minute: int = 100,
):
    """Generate from OpenAI Chat Completion API.

    Args:
        messages: List of messages to proceed.
        engine_name: Engine name to use, see https://platform.openai.com/docs/models
        temperature: Temperature to use.
        max_tokens: Maximum number of tokens to generate.
        top_p: Top p to use.
        requests_per_minute: Number of requests per minute to allow.

    Returns:
        List of generated responses.
    """    
    # https://chat.openai.com/share/09154613-5f66-4c74-828b-7bd9384c2168
    delay = 60.0 / requests_per_minute
    limiter = aiolimiter.AsyncLimiter(1, delay)
    async_responses = [
        _throttled_openai_chat_completion_acreate(
            client,
            model=engine_name,
            messages=message,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            limiter=limiter,
        )
        for message in messages
    ]
    
    responses = await tqdm_asyncio.gather(*async_responses)
    
    outputs = []
    for response in responses:
        if response:
            outputs.append(response.choices[0].message.content)
        else:
            outputs.append("Invalid Message")
    return outputs