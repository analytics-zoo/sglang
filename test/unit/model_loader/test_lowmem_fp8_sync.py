"""
Unit test for LowMemFp8ModelLoader synchronization fix.

Verifies that current_platform.synchronize() is called BEFORE empty_cache()
to prevent hangs on XPU during FP8 weight finalization for large MoE models.
"""

import gc
from unittest.mock import MagicMock, patch, call
import torch
import torch.nn as nn
import pytest


class MockQuantMethod:
    """Mock quant_method for testing."""
    def process_weights_after_loading(self, module):
        pass


class SimpleModule(nn.Module):
    """Simple module with quant_method for testing."""
    def __init__(self, has_quant=True):
        super().__init__()
        self.linear = nn.Linear(10, 10)
        if has_quant:
            self.quant_method = MockQuantMethod()


class ManyModulesModel(nn.Module):
    """Model with many modules to trigger periodic sync/cache clear."""
    def __init__(self, num_modules=20):
        super().__init__()
        # Create enough modules to trigger the periodic sync (every 8 modules)
        self.layers = nn.ModuleList([
            SimpleModule(has_quant=True) for _ in range(num_modules)
        ])


def test_synchronize_called_before_empty_cache():
    """
    Test that synchronize() is always called before empty_cache().
    
    This is critical for XPU: without synchronize(), async operations can pile up
    causing hangs on large MoE models during FP8 weight finalization.
    """
    # Import the loader class and function
    from sglang.srt.model_loader.loader import LowMemFp8ModelLoader
    
    # Create model with enough modules to trigger periodic sync (>8)
    model = ManyModulesModel(num_modules=20)
    target_device = torch.device("cpu")  # Use CPU for testing
    
    call_order = []
    
    def mock_synchronize():
        call_order.append('synchronize')
    
    def mock_empty_cache():
        call_order.append('empty_cache')
    
    # Patch current_platform's synchronize and empty_cache
    with patch('sglang.srt.model_loader.loader.current_platform') as mock_platform:
        mock_platform.synchronize = mock_synchronize
        mock_platform.empty_cache = mock_empty_cache
        
        # Also patch gc.collect to track order
        with patch('sglang.srt.model_loader.loader.gc.collect') as mock_gc:
            mock_gc.side_effect = lambda: call_order.append('gc_collect')
            
            # Call the method under test
            LowMemFp8ModelLoader._move_and_quantize_per_module(model, target_device)
    
    # Verify synchronize is called before empty_cache
    # Filter to just sync and cache calls
    sync_cache_calls = [c for c in call_order if c in ('synchronize', 'empty_cache')]
    
    # There should be at least one sync+cache pair
    assert len(sync_cache_calls) >= 2, f"Expected sync+cache calls, got: {sync_cache_calls}"
    
    # Every empty_cache must be preceded by synchronize
    for i, c in enumerate(sync_cache_calls):
        if c == 'empty_cache':
            assert i > 0, "empty_cache called without prior synchronize"
            assert sync_cache_calls[i-1] == 'synchronize', \
                f"empty_cache at index {i} not preceded by synchronize. Order: {sync_cache_calls}"
    
    print(f"PASS: Call order verified: {sync_cache_calls}")


def test_sync_called_periodically_and_at_end():
    """
    Test that sync is called both periodically (every 8 modules) AND at the end.
    """
    from sglang.srt.model_loader.loader import LowMemFp8ModelLoader
    
    # Create model with 20 modules with quant_methods
    # This should trigger: 1 sync at module 8, 1 sync at module 16, 1 final sync
    model = ManyModulesModel(num_modules=20)
    target_device = torch.device("cpu")
    
    sync_count = [0]
    empty_cache_count = [0]
    
    with patch('sglang.srt.model_loader.loader.current_platform') as mock_platform:
        mock_platform.synchronize = lambda: sync_count.__setitem__(0, sync_count[0] + 1)
        mock_platform.empty_cache = lambda: empty_cache_count.__setitem__(0, empty_cache_count[0] + 1)
        
        with patch('sglang.srt.model_loader.loader.gc.collect'):
            LowMemFp8ModelLoader._move_and_quantize_per_module(model, target_device)
    
    # Should have at least 3 syncs: at module 8, 16, and final
    assert sync_count[0] >= 3, f"Expected at least 3 sync calls, got {sync_count[0]}"
    assert sync_count[0] == empty_cache_count[0], \
        f"sync count ({sync_count[0]}) should equal empty_cache count ({empty_cache_count[0]})"
    
    print(f"PASS: sync called {sync_count[0]} times, empty_cache called {empty_cache_count[0]} times")


if __name__ == "__main__":
    test_synchronize_called_before_empty_cache()
    test_sync_called_periodically_and_at_end()
    print("\nAll tests passed!")
