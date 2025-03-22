import time
import threading
from datetime import datetime

class PriceTracker:
    """
    Real-time price tracking for cryptocurrencies
    """
    
    def __init__(self, api_client):
        """
        Initialize the price tracker
        
        Args:
            api_client: API client for fetching market data
        """
        self.api_client = api_client
        self.running = False
        self.thread = None
        self.price_history = {}
        self.latest_prices = {}
        self.update_frequency = 5  # seconds
        self._callbacks = []
    
    def start_tracking(self, symbols=["BTCUSDT"]):
        """
        Start tracking prices for given symbols
        
        Args:
            symbols: List of symbols to track
        """
        if self.running:
            return
            
        self.symbols = symbols
        self.running = True
        
        # Initialize empty price history
        for symbol in symbols:
            self.price_history[symbol] = []
        
        self.thread = threading.Thread(target=self._tracking_loop)
        self.thread.daemon = True
        self.thread.start()
        
        return True
    
    def stop_tracking(self):
        """Stop price tracking"""
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        return True
    
    def _tracking_loop(self):
        """Main tracking loop"""
        while self.running:
            for symbol in self.symbols:
                try:
                    current_price = self.api_client.get_market_price(symbol)
                    
                    if not isinstance(current_price, (int, float)):
                        continue
                    
                    # Update latest price and history
                    timestamp = datetime.now()
                    self.latest_prices[symbol] = {
                        'price': current_price,
                        'timestamp': timestamp
                    }
                    
                    self.price_history[symbol].append({
                        'price': current_price,
                        'timestamp': timestamp
                    })
                    
                    # Keep history limited to last 1000 data points
                    if len(self.price_history[symbol]) > 1000:
                        self.price_history[symbol] = self.price_history[symbol][-1000:]
                    
                    # Execute callbacks with new price data
                    for callback in self._callbacks:
                        try:
                            callback(symbol, current_price, timestamp)
                        except Exception as e:
                            print(f"Error in price callback: {str(e)}")
                    
                except Exception as e:
                    print(f"Error tracking price for {symbol}: {str(e)}")
            
            # Sleep to avoid API rate limits
            time.sleep(self.update_frequency)
    
    def get_latest_price(self, symbol):
        """
        Get latest price for a symbol
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            Dictionary with price and timestamp
        """
        return self.latest_prices.get(symbol, None)
    
    def get_price_history(self, symbol, limit=100):
        """
        Get price history for a symbol
        
        Args:
            symbol: Trading pair symbol
            limit: Maximum number of data points to return
            
        Returns:
            List of price data points
        """
        history = self.price_history.get(symbol, [])
        if limit:
            return history[-limit:]
        return history
    
    def add_price_callback(self, callback):
        """
        Add callback function for price updates
        
        Args:
            callback: Function to call with (symbol, price, timestamp)
        """
        if callback not in self._callbacks:
            self._callbacks.append(callback)
    
    def remove_price_callback(self, callback):
        """
        Remove price update callback
        
        Args:
            callback: Function to remove
        """
        if callback in self._callbacks:
            self._callbacks.remove(callback)