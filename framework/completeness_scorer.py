"""
Completeness Scorer Module
Evaluates the completeness and reasoning quality of model responses.
"""

import re

class CompletenessScorer:
    """
    Scores the completeness of model responses.
    
    Evaluates:
    - Presence of step-by-step reasoning
    - Explanation quality
    - Structure and organization
    - Sufficient detail
    """
    
    def score(self, response: str, problem: str = None) -> float:
        """
        Score the completeness of a response.
        
        Args:
            response: Model's response
            problem: Original problem (for context)
            
        Returns:
            Completeness score between 0 and 1
        """
        if not response or len(response.strip()) == 0:
            return 0.0
        
        score = 0.0
        
        # Component 1: Length and detail (max 0.25)
        score += self._score_length(response)
        
        # Component 2: Step-by-step reasoning (max 0.30)
        score += self._score_steps(response)
        
        # Component 3: Structure and organization (max 0.25)
        score += self._score_structure(response)
        
        # Component 4: Explanation quality (max 0.20)
        score += self._score_explanation(response)
        
        return min(score, 1.0)
    
    def _score_length(self, response: str) -> float:
        """
        Score based on response length and detail.
        
        Returns: 0.0 to 0.25
        """
        word_count = len(response.split())
        
        if word_count >= 100:
            return 0.25
        elif word_count >= 50:
            return 0.20
        elif word_count >= 30:
            return 0.15
        elif word_count >= 15:
            return 0.10
        elif word_count >= 5:
            return 0.05
        return 0.0
    
    def _score_steps(self, response: str) -> float:
        """
        Score based on presence of step-by-step reasoning.
        
        Looks for:
        - Numbered or bulleted steps
        - Sequential reasoning indicators
        - Multiple calculation stages
        
        Returns: 0.0 to 0.30
        """
        score = 0.0
        
        # Check for numbered steps (1., 2., Step 1, etc.)
        step_patterns = [
            r'\b(?:step\s+)?\d+[\.\):]',  # "Step 1:", "1.", "1)"
            r'\b(?:first|second|third|next|then|finally)\b',  # Sequential words
            r'^\s*[-*•]\s',  # Bullet points
        ]
        
        step_count = 0
        for pattern in step_patterns:
            matches = re.findall(pattern, response, re.IGNORECASE | re.MULTILINE)
            step_count += len(matches)
        
        if step_count >= 5:
            score += 0.30
        elif step_count >= 3:
            score += 0.20
        elif step_count >= 1:
            score += 0.10
        
        return score
    
    def _score_structure(self, response: str) -> float:
        """
        Score based on structure and organization.
        
        Checks for:
        - Multiple paragraphs
        - Clear sections
        - Final answer section
        
        Returns: 0.0 to 0.25
        """
        score = 0.0
        
        # Multiple paragraphs/sections
        paragraph_count = len([p for p in response.split('\n\n') if p.strip()])
        line_count = len([l for l in response.split('\n') if l.strip()])
        
        if paragraph_count >= 3 or line_count >= 5:
            score += 0.10
        elif paragraph_count >= 2 or line_count >= 3:
            score += 0.05
        
        # Has clear answer section
        if re.search(r'(?:answer|result|solution|final|therefore)\s*[:\-=]', 
                    response, re.IGNORECASE):
            score += 0.10
        
        # Has conclusion indicators
        if any(word in response.lower() for word in 
               ['therefore', 'thus', 'hence', 'so', 'conclusion', 'finally']):
            score += 0.05
        
        return min(score, 0.25)
    
    def _score_explanation(self, response: str) -> float:
        """
        Score explanation quality.
        
        Checks for:
        - Reasoning words (because, since, as, etc.)
        - Mathematical operations explained
        - Context and interpretation
        
        Returns: 0.0 to 0.20
        """
        score = 0.0
        
        # Reasoning words
        reasoning_words = ['because', 'since', 'as', 'when', 'where', 'why', 
                          'this means', 'which gives', 'we get', 'we can']
        reasoning_count = sum(1 for word in reasoning_words if word in response.lower())
        
        if reasoning_count >= 3:
            score += 0.10
        elif reasoning_count >= 1:
            score += 0.05
        
        # Mathematical operations explained
        operation_words = ['add', 'subtract', 'multiply', 'divide', 'calculate', 
                          'compute', 'solve', 'equals', 'simplify']
        operation_count = sum(1 for word in operation_words if word in response.lower())
        
        if operation_count >= 2:
            score += 0.10
        elif operation_count >= 1:
            score += 0.05
        
        return min(score, 0.20)
