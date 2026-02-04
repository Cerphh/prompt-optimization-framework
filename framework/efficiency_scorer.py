"""
Efficiency Scorer Module
Evaluates the efficiency of model responses based on latency and token usage.
"""

class EfficiencyScorer:
    """
    Scores the efficiency of model responses.
    
    Considers:
    - Response latency (time taken)
    - Token usage (prompt + completion tokens)
    - Response conciseness
    """
    
    def score(self, response: str, metrics: dict = None) -> float:
        """
        Score the efficiency of a response.
        
        Args:
            response: Model's response text
            metrics: Dictionary with 'elapsed_time', 'total_tokens', etc.
            
        Returns:
            Efficiency score between 0 and 1
        """
        if metrics is None:
            metrics = {}
        
        elapsed_time = metrics.get('elapsed_time', 0)
        total_tokens = metrics.get('total_tokens', 0)
        
        # Component 1: Time efficiency (40%)
        time_score = self._score_time(elapsed_time)
        
        # Component 2: Token efficiency (30%)
        token_score = self._score_tokens(total_tokens, response)
        
        # Component 3: Conciseness (30%)
        conciseness_score = self._score_conciseness(response)
        
        # Weighted combination
        overall_score = (time_score * 0.4 + 
                        token_score * 0.3 + 
                        conciseness_score * 0.3)
        
        return round(overall_score, 3)
    
    def _score_time(self, elapsed_time: float) -> float:
        """
        Score based on response latency.
        
        Optimal range: 2-10 seconds
        Too fast might indicate incomplete answer
        Too slow is inefficient
        
        Returns: 0.0 to 1.0
        """
        if elapsed_time == 0:
            return 0.5  # Neutral if no timing data
        
        if elapsed_time < 1:
            return 0.6  # Too fast, might be incomplete
        elif elapsed_time <= 3:
            return 1.0  # Excellent
        elif elapsed_time <= 8:
            return 0.9  # Very good
        elif elapsed_time <= 15:
            return 0.7  # Acceptable
        elif elapsed_time <= 30:
            return 0.5  # Slow
        else:
            return 0.3  # Too slow
    
    def _score_tokens(self, total_tokens: int, response: str) -> float:
        """
        Score based on token usage efficiency.
        
        Considers total tokens used and tokens per word ratio.
        
        Returns: 0.0 to 1.0
        """
        if total_tokens == 0:
            return 0.5  # Neutral if no token data
        
        # Score based on total token count
        if total_tokens <= 100:
            token_count_score = 1.0
        elif total_tokens <= 300:
            token_count_score = 0.9
        elif total_tokens <= 500:
            token_count_score = 0.7
        elif total_tokens <= 1000:
            token_count_score = 0.5
        else:
            token_count_score = 0.3
        
        # Check tokens-to-words ratio (should be reasonable)
        word_count = len(response.split())
        if word_count > 0:
            ratio = total_tokens / word_count
            # Typical ratio is 1.3-1.5 tokens per word
            if 1.0 <= ratio <= 2.0:
                ratio_score = 1.0
            elif 0.8 <= ratio <= 2.5:
                ratio_score = 0.8
            else:
                ratio_score = 0.6
        else:
            ratio_score = 0.5
        
        return (token_count_score + ratio_score) / 2
    
    def _score_conciseness(self, response: str) -> float:
        """
        Score response conciseness.
        
        Balance between being thorough and being concise.
        Ideal: 30-150 words for math problems
        
        Returns: 0.0 to 1.0
        """
        word_count = len(response.split())
        
        if 30 <= word_count <= 100:
            return 1.0  # Optimal
        elif 20 <= word_count < 30:
            return 0.9  # Slightly terse
        elif 100 < word_count <= 150:
            return 0.9  # Slightly verbose
        elif 10 <= word_count < 20:
            return 0.7  # Too terse
        elif 150 < word_count <= 250:
            return 0.7  # Verbose
        elif word_count < 10:
            return 0.5  # Way too terse
        else:
            return 0.5  # Way too verbose
