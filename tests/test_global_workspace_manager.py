import os
import sys
import unittest
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import jax
import jax.numpy as jnp

from aura.consciousness.global_workspace_manager import (
    GlobalWorkspaceManager, 
    GlobalWorkspaceConfig,
    ConsciousContent
)


class TestGlobalWorkspaceManager(unittest.TestCase):
    
    def test_consciousness_competition(self):
        """Test consciousness competition with scoring and thresholding."""
        config = GlobalWorkspaceConfig(capacity=3, threshold=0.5)
        manager = GlobalWorkspaceManager(config)
        
        # Create candidates with different scores
        candidates = [
            ConsciousContent(
                id="content_1",
                text="High relevance content",
                embedding=jnp.ones(10),
                timestamp=time.time(),
                relevance=0.8,
                novelty=0.6,
                emotional_salience=0.7,
                strength=1.0
            ),
            ConsciousContent(
                id="content_2",
                text="Low relevance content",
                embedding=jnp.ones(10),
                timestamp=time.time(),
                relevance=0.2,
                novelty=0.3,
                emotional_salience=0.1,
                strength=0.5
            ),
            ConsciousContent(
                id="content_3",
                text="Medium relevance content",
                embedding=jnp.ones(10),
                timestamp=time.time(),
                relevance=0.6,
                novelty=0.7,
                emotional_salience=0.5,
                strength=0.8
            )
        ]
        
        winners = manager.compete_for_consciousness(candidates)
        
        # Should have 2 winners (content_1 and content_3) above threshold
        # content_2 should be below threshold
        self.assertEqual(len(winners), 2)
        
        # Winners should be the high-scoring contents
        winner_ids = [winner.id for winner in winners]
        self.assertIn("content_1", winner_ids)
        self.assertIn("content_3", winner_ids)
        self.assertNotIn("content_2", winner_ids)
        
        # Workspace should contain winners
        self.assertEqual(len(manager.workspace), 2)
    
    def test_workspace_capacity_limit(self):
        """Test that workspace respects capacity limits."""
        config = GlobalWorkspaceConfig(capacity=2, threshold=0.1)
        manager = GlobalWorkspaceManager(config)
        
        # Create more candidates than capacity
        candidates = [
            ConsciousContent(
                id=f"content_{i}",
                text=f"Content {i}",
                embedding=jnp.ones(10),
                timestamp=time.time(),
                relevance=0.9,
                novelty=0.8,
                emotional_salience=0.7,
                strength=1.0
            )
            for i in range(5)  # 5 candidates
        ]
        
        winners = manager.compete_for_consciousness(candidates)
        
        # Should only have 2 winners despite 5 candidates
        self.assertEqual(len(winners), 2)
        self.assertEqual(len(manager.workspace), 2)
    
    def test_workspace_status(self):
        """Test workspace status reporting."""
        config = GlobalWorkspaceConfig(capacity=3, threshold=0.5)
        manager = GlobalWorkspaceManager(config)
        
        content = ConsciousContent(
            id="test_content",
            text="Test content",
            embedding=jnp.ones(10),
            timestamp=time.time(),
            relevance=0.8,
            novelty=0.6,
            emotional_salience=0.7,
            strength=1.0
        )
        
        manager.compete_for_consciousness([content])
        status = manager.get_workspace_status()
        
        # Check status structure
        self.assertIn('current_size', status)
        self.assertIn('capacity', status)
        self.assertIn('consciousness_threshold', status)
        self.assertIn('contents', status)
        
        # Check values
        self.assertEqual(status['current_size'], 1)
        self.assertEqual(status['capacity'], 3)
        self.assertEqual(status['consciousness_threshold'], 0.5)
        
        # Check contents
        self.assertEqual(len(status['contents']), 1)
        self.assertEqual(status['contents'][0]['id'], 'test_content')
        self.assertEqual(status['contents'][0]['text'], 'Test content')
    
    def test_clear_workspace(self):
        """Test clearing workspace contents."""
        config = GlobalWorkspaceConfig(capacity=3, threshold=0.1)
        manager = GlobalWorkspaceManager(config)
        
        content = ConsciousContent(
            id="test_content",
            text="Test content",
            embedding=jnp.ones(10),
            timestamp=time.time(),
            relevance=0.8,
            novelty=0.6,
            emotional_salience=0.7,
            strength=1.0
        )
        
        manager.compete_for_consciousness([content])
        self.assertEqual(len(manager.workspace), 1)
        
        manager.clear_workspace()
        self.assertEqual(len(manager.workspace), 0)
    
    def test_remove_oldest_content(self):
        """Test removing oldest content from workspace."""
        config = GlobalWorkspaceConfig(capacity=3, threshold=0.1)
        manager = GlobalWorkspaceManager(config)
        
        # Create contents with different timestamps
        old_content = ConsciousContent(
            id="old_content",
            text="Old content",
            embedding=jnp.ones(10),
            timestamp=time.time() - 100,  # Older
            relevance=0.8,
            novelty=0.6,
            emotional_salience=0.7,
            strength=1.0
        )
        
        new_content = ConsciousContent(
            id="new_content",
            text="New content",
            embedding=jnp.ones(10),
            timestamp=time.time(),  # Newer
            relevance=0.8,
            novelty=0.6,
            emotional_salience=0.7,
            strength=1.0
        )
        
        manager.compete_for_consciousness([old_content, new_content])
        self.assertEqual(len(manager.workspace), 2)
        
        # Remove oldest content
        manager.remove_oldest_content()
        
        # Should only have new content left
        self.assertEqual(len(manager.workspace), 1)
        self.assertEqual(manager.workspace[0].id, "new_content")


if __name__ == '__main__':
    unittest.main()
