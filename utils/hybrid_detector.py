import numpy as np
import logging
from typing import Dict, Tuple

logger = logging.getLogger(__name__)

class HybridDetector:
    def __init__(self, ml_threshold=0.8):
        self.ml_threshold = ml_threshold
        self.ml_detector = None
        self.conventional_detector = None
    
    def detect_deadlock(self, system_state: Dict) -> Tuple[bool, Dict]:
        """
        Hybrid deadlock detection using both ML and conventional methods
        """
        # Lazy initialization to avoid circular imports
        if self.ml_detector is None:
            from .ml_detector import MLDetector
            self.ml_detector = MLDetector()
        
        if self.conventional_detector is None:
            from .conventional_detector import ConventionalDetector
            self.conventional_detector = ConventionalDetector()
        
        # First, use ML for quick prediction
        ml_deadlock, ml_details = self.ml_detector.detect_deadlock(system_state, "Random Forest")
        confidence = ml_details.get("confidence", 0.5)
        
        # If ML is very confident, return its result
        if confidence > self.ml_threshold:
            return ml_deadlock, {
                "method": "ML",
                "confidence": confidence,
                "ml_details": ml_details
            }
        
        # If ML is not confident, use conventional method
        conventional_deadlock, conventional_details = self.conventional_detector.detect_deadlock(
            system_state["processes"], system_state["available"], "Wait-for Graph"
        )
        
        return conventional_deadlock, {
            "method": "Conventional",
            "confidence": 1.0,
            "conventional_details": conventional_details,
            "ml_confidence": confidence
        }
    
    def adaptive_threshold(self, historical_accuracy: Dict) -> float:
        """
        Adjust the ML confidence threshold based on historical performance
        """
        if not historical_accuracy:
            return self.ml_threshold
        
        # Calculate average ML accuracy
        ml_accuracy = historical_accuracy.get("Random Forest", 0.5)
        
        # Adjust threshold based on accuracy
        # Higher accuracy -> lower threshold (more trust in ML)
        # Lower accuracy -> higher threshold (less trust in ML)
        if ml_accuracy > 0.9:
            return 0.7
        elif ml_accuracy > 0.8:
            return 0.75
        elif ml_accuracy > 0.7:
            return 0.8
        else:
            return 0.85