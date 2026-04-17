"""Text cleaning utilities for social media data.

Provides the TweetCleaner class for preprocessing raw tweet text by removing
URLs, mentions, hashtags, special characters, and normalizing whitespace.
"""

from __future__ import annotations

import re


class TweetCleaner:
    """Cleaner for preprocessing social media text.

    Provides methods to remove common noise from tweets including URLs,
    user mentions, hashtags, and special characters, plus whitespace
    normalization.
    """

    def remove_urls(self, text: str) -> str:
        """Remove HTTP/HTTPS URLs and www links from text.

        Args:
            text: Raw input string potentially containing URLs.

        Returns:
            String with URLs removed.
        """
        return re.sub(r"https?://\S+|www\.\S+", "", text)

    def remove_mentions(self, text: str) -> str:
        """Remove @username mentions from text.

        Args:
            text: Raw input string potentially containing @mentions.

        Returns:
            String with mentions removed.
        """
        return re.sub(r"@\w+", "", text)

    def remove_hashtags(self, text: str) -> str:
        """Remove entire hashtags (including word) from text.

        Args:
            text: Raw input string potentially containing #hashtags.

        Returns:
            String with hashtags removed.
        """
        return re.sub(r"#\w+", "", text)

    def remove_special_chars(self, text: str) -> str:
        """Remove non-alphanumeric characters except whitespace.

        Args:
            text: Raw input string with potential special characters.

        Returns:
            String with only alphanumeric characters and whitespace.
        """
        return re.sub(r"[^a-zA-Z0-9\s]", "", text)

    def normalize_whitespace(self, text: str) -> str:
        """Collapse multiple whitespace characters and strip leading/trailing.

        Args:
            text: Input string with potential extra whitespace.

        Returns:
            String with normalized single spaces and no leading/trailing whitespace.
        """
        return re.sub(r"\s+", " ", text).strip()
