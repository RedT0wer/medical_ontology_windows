"""Тесты для модуля вычисления векторных эмбеддингов src/embeddings.py."""

import unittest
from unittest.mock import MagicMock, patch
import numpy as np
from src.embeddings import (
    BaseEmbeddingProvider,
    FastEmbedProvider,
    OllamaEmbeddingProvider,
    OpenAIEmbeddingProvider,
    get_embedding_provider,
    l2_normalize,
)


class TestEmbeddings(unittest.TestCase):
    """Проверка нормализации векторов и провайдеров эмбеддингов."""

    def test_l2_normalize_normal_vector(self):
        """Проверка нормализации ненулевого вектора."""
        vec = [3.0, 4.0]
        normalized = l2_normalize(vec)
        self.assertAlmostEqual(normalized[0], 0.6)
        self.assertAlmostEqual(normalized[1], 0.8)
        norm = np.linalg.norm(np.array(normalized))
        self.assertAlmostEqual(norm, 1.0, places=5)

    def test_l2_normalize_zero_vector(self):
        """Проверка нормализации нулевого вектора (без деления на ноль)."""
        vec = [0.0, 0.0, 0.0]
        normalized = l2_normalize(vec)
        self.assertEqual(normalized, [0.0, 0.0, 0.0])

    @patch("src.embeddings.FastEmbedProvider.__init__", return_value=None)
    def test_fastembed_provider(self, mock_init):
        """Проверка работы FastEmbedProvider с замоканной моделью."""
        provider = FastEmbedProvider("test-model")
        mock_model = MagicMock()
        mock_model.embed.return_value = [np.array([3.0, 4.0])]
        provider.model = mock_model

        res = provider.embed_text("печень")
        self.assertAlmostEqual(res[0], 0.6)
        self.assertAlmostEqual(res[1], 0.8)

        mock_model.embed.return_value = [np.array([1.0, 0.0]), np.array([0.0, 2.0])]
        batch_res = provider.embed_batch(["почка", "селезенка"])
        self.assertEqual(len(batch_res), 2)
        self.assertAlmostEqual(batch_res[0][0], 1.0)
        self.assertAlmostEqual(batch_res[1][1], 1.0)

    @patch("requests.post")
    def test_ollama_provider_success(self, mock_post):
        """Проверка OllamaEmbeddingProvider при успешном ответе."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"embeddings": [[3.0, 4.0]]}
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        provider = OllamaEmbeddingProvider(base_url="http://127.0.0.1:11434", model_name="nomic-embed-text")
        emb = provider.embed_text("сердце")

        self.assertAlmostEqual(emb[0], 0.6)
        self.assertAlmostEqual(emb[1], 0.8)
        mock_post.assert_called_once()

    @patch("requests.post")
    def test_ollama_provider_error(self, mock_post):
        """Проверка обработки ошибок сетью в OllamaEmbeddingProvider."""
        mock_post.side_effect = Exception("Ollama connection failed")
        provider = OllamaEmbeddingProvider(base_url="http://127.0.0.1:11434", model_name="nomic-embed-text")

        with self.assertRaises(Exception):
            provider.embed_text("тест")

    @patch("openai.OpenAI")
    def test_openai_provider(self, mock_openai_cls):
        """Проверка OpenAIEmbeddingProvider."""
        mock_client = MagicMock()
        mock_item = MagicMock()
        mock_item.embedding = [3.0, 4.0]
        mock_res = MagicMock()
        mock_res.data = [mock_item]
        mock_client.embeddings.create.return_value = mock_res
        mock_openai_cls.return_value = mock_client

        provider = OpenAIEmbeddingProvider(model_name="text-embedding-3-small", api_key="sk-test")
        emb = provider.embed_text("легкие")

        self.assertAlmostEqual(emb[0], 0.6)
        self.assertAlmostEqual(emb[1], 0.8)

    def test_get_embedding_provider_factory(self):
        """Проверка фабрики get_embedding_provider."""
        import src.embeddings as emb_mod

        # Сбрасываем кэш синглтона
        emb_mod._embedding_service = None

        with patch("src.embeddings.settings.EMBEDDING_PROVIDER", "unknown_provider"):
            with self.assertRaises(ValueError):
                get_embedding_provider()

        # Тест синглтона
        mock_dummy = MagicMock(spec=BaseEmbeddingProvider)
        emb_mod._embedding_service = mock_dummy
        self.assertIs(get_embedding_provider(), mock_dummy)
        emb_mod._embedding_service = None


if __name__ == "__main__":
    unittest.main()
