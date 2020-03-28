# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, you can obtain one at http://mozilla.org/MPL/2.0/.

# Modified from https://github.com/mozilla/gcp-ingestion/blob/master/ingestion-edge/ingestion_edge/util.py  # noqa

"""Utilities."""

from google.cloud.pubsub_v1.gapic.publisher_client import PublisherClient
from google.cloud.pubsub_v1.publisher.exceptions import PublishError
from google.cloud.pubsub_v1.types import BatchSettings, PublishRequest, PubsubMessage
from typing import List, Optional
import asyncio


class AsyncioBatch:
    """Batch for google.cloud.pubsub_v1.PublisherClient in asyncio.

    Workaround for https://github.com/googleapis/google-cloud-python/issues/7104
    """

    def __init__(
        self,
        client: PublisherClient,
        topic: str,
        settings: BatchSettings,
        autocommit: bool = True,
        **kwargs
    ):
        print(kwargs)

        """Initialize."""
        self.client = client
        self.topic = topic
        self.settings = settings

        self.full = asyncio.Event()
        self.messages: List[PubsubMessage] = []
        # fix https://github.com/googleapis/google-cloud-python/issues/7108
        self.size = PublishRequest(topic=topic, messages=[]).ByteSize()

        # Create a task to commit when full
        self.result = asyncio.create_task(self.commit())

        # If max latency is specified start a task to monitor the batch
        # and commit when max latency is reached
        if autocommit and self.settings.max_latency < float("inf"):
            asyncio.create_task(self.monitor())

    async def monitor(self):
        """Sleep until max latency is reached then set the batch to full."""
        await asyncio.sleep(self.settings.max_latency)
        self.full.set()

    async def commit(self) -> List[str]:
        """Publish this batch when full."""
        await self.full.wait()

        if not self.messages:
            return []

        response = await asyncio.get_running_loop().run_in_executor(
            None, self.client.api.publish, self.topic, self.messages
        )

        if len(response.message_ids) != len(self.messages):
            raise PublishError("Some messages not successfully published")

        return response.message_ids

    def publish(self, message: PubsubMessage) -> Optional[asyncio.Task]:
        """Asynchronously publish message."""
        if not self.full.is_set():
            # Check if batch has space to accept message
            new_size = self.size + PublishRequest(messages=[message]).ByteSize()
            if new_size <= self.settings.max_bytes:
                # Store message in the batch
                index = len(self.messages)
                self.messages.append(message)
                self.size = new_size

                # Check if batch is now full
                full = (
                    new_size >= self.settings.max_bytes
                    or index + 1 >= self.settings.max_messages
                )
                if full:
                    self.full.set()

                # Return a task to await for the message id
                return asyncio.create_task(self.message_id(index))
            elif not self.messages:
                # Fix https://github.com/googleapis/google-cloud-python/issues/7107
                raise ValueError("Message exceeds max bytes")

        return None  # the batch cannot accept a message

    async def message_id(self, index: int) -> str:
        """Get message id from result by index."""
        return (await self.result)[index]
