import requests
import json
import time
import hmac
import hashlib
from datetime import datetime
from urllib.parse import urlencode

class RoostooAPI:
    """
    A wrapper for the Roostoo Trading API based on official documentation
    """
    
    def __init__(self, api_key=None, secret_key=None):
        """
        Initialize the Roostoo API wrapper
        
        Args:
            api_key: Roostoo API key
            secret_key: Roostoo Secret key for signing requests
        """
        self.api_key = api_key
        self.secret_key = secret_key
        self.base_url = "https://mock-api.roostoo.com"  # Using the mock API URL from docs
        self.headers = {
            "Content-Type": "application/x-www-form-urlencoded"
        }
        
        if self.api_key:
            self.headers["RST-API-KEY"] = self.api_key
    
    def set_api_key(self, api_key, secret_key=None):
        """
        Set or update the API key
        
        Args:
            api_key: Roostoo API key
            secret_key: Roostoo Secret key for signing requests
        """
        self.api_key = api_key
        if secret_key:
            self.secret_key = secret_key
        
        self.headers["RST-API-KEY"] = self.api_key
    
    def _get_timestamp(self):
        """Get current timestamp in milliseconds"""
        return int(time.time() * 1000)
    
    def _sign_request(self, params):
        """
        Sign request parameters with HMAC SHA256
        
        Args:
            params: Dictionary of request parameters
            
        Returns:
            Signature string
        """
        if not self.secret_key:
            return None
            
        # Sort parameters by key and create query string
        sorted_params = sorted(params.items())
        query_string = '&'.join([f"{key}={value}" for key, value in sorted_params])
        
        # Create HMAC SHA256 signature
        signature = hmac.new(
            self.secret_key.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
        return signature
    
    def _add_signature_header(self, params):
        """
        Add signature to request headers
        
        Args:
            params: Dictionary of request parameters
            
        Returns:
            Updated headers dictionary
        """
        signature = self._sign_request(params)
        headers = self.headers.copy()
        
        if signature:
            headers["MSG-SIGNATURE"] = signature
            
        if self.api_key:
            headers["RST-API-KEY"] = self.api_key
            
        return headers
    
    def test_connection(self):
        """
        Test the API connection using server time endpoint
        
        Returns:
            Boolean indicating success or failure
        """
        try:
            response = requests.get(
                f"{self.base_url}/v3/serverTime", 
                timeout=10
            )
            return response.status_code == 200 and "ServerTime" in response.json()
        except:
            return False
    
    def get_server_time(self):
        """
        Get server time
        
        Returns:
            Server time as timestamp
        """
        try:
            response = requests.get(
                f"{self.base_url}/v3/serverTime"
            )
            data = response.json()
            return data.get("ServerTime")
        except Exception as e:
            return {"error": str(e)}
            
    def get_exchange_info(self):
        """
        Get exchange information
        
        Returns:
            Dictionary with exchange information
        """
        try:
            response = requests.get(
                f"{self.base_url}/v3/exchangeInfo"
            )
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def get_market_ticker(self, symbol=None):
        """
        Get market ticker information
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USD')
            
        Returns:
            Dictionary with ticker information
        """
        params = {
            "timestamp": self._get_timestamp()
        }
        
        if symbol:
            params["pair"] = symbol
        
        try:
            response = requests.get(
                f"{self.base_url}/v3/ticker", 
                params=params
            )
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def get_account_balance(self):
        """
        Get account balance information
        
        Returns:
            Dictionary with balance information
        """
        params = {
            "timestamp": self._get_timestamp()
        }
        
        headers = self._add_signature_header(params)
        
        try:
            response = requests.get(
                f"{self.base_url}/v3/balance", 
                headers=headers,
                params=params
            )
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def get_pending_orders_count(self):
        """
        Get number of pending orders
        
        Returns:
            Dictionary with pending orders count
        """
        params = {
            "timestamp": self._get_timestamp()
        }
        
        headers = self._add_signature_header(params)
        
        try:
            response = requests.get(
                f"{self.base_url}/v3/pending_count", 
                headers=headers,
                params=params
            )
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def place_order(self, symbol, side, quantity, order_type="MARKET", price=None):
        """
        Place a trading order
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USD')
            side: Order side ('BUY' or 'SELL')
            quantity: Order quantity
            order_type: Order type ('MARKET' or 'LIMIT')
            price: Limit price (required for LIMIT orders)
            
        Returns:
            Order details
        """
        params = {
            "pair": symbol,
            "side": side,
            "type": order_type,
            "quantity": str(quantity),
            "timestamp": str(self._get_timestamp())
        }
        
        if order_type == "LIMIT" and price is not None:
            params["price"] = str(price)
        
        headers = self._add_signature_header(params)
        headers["Content-Type"] = "application/x-www-form-urlencoded"
        
        try:
            response = requests.post(
                f"{self.base_url}/v3/place_order", 
                headers=headers,
                data=params
            )
            
            if response.status_code == 200:
                try:
                    return response.json()
                except ValueError:
                    return {"Success": False, "ErrMsg": f"Invalid JSON response: {response.text}"}
            else:
                return {"Success": False, "ErrMsg": f"HTTP {response.status_code}: {response.text}"}
                
        except Exception as e:
            return {"Success": False, "ErrMsg": f"Request error: {str(e)}"}
    
    def query_order(self, order_id=None, symbol=None, limit=100, pending_only=False):
        """
        Query order information
        
        Args:
            order_id: Order ID
            symbol: Trading pair symbol
            limit: Maximum number of orders to return
            pending_only: Whether to return only pending orders
            
        Returns:
            Dictionary with order information
        """
        params = {
            "timestamp": self._get_timestamp()
        }
        
        if order_id:
            params["order_id"] = str(order_id)
        elif symbol:
            params["pair"] = symbol
            
        if limit:
            params["limit"] = str(limit)
            
        if pending_only:
            params["pending_only"] = "TRUE"
        
        headers = self._add_signature_header(params)
        
        try:
            response = requests.post(
                f"{self.base_url}/v3/query_order", 
                headers=headers,
                data=params
            )
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def cancel_order(self, order_id):
        """
        Cancel an order
        
        Args:
            order_id: Order ID
            
        Returns:
            Cancellation status
        """
        params = {
            "order_id": str(order_id),
            "timestamp": self._get_timestamp()
        }
        
        headers = self._add_signature_header(params)
        
        try:
            response = requests.post(
                f"{self.base_url}/v3/cancel_order", 
                headers=headers,
                data=params
            )
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    # Convenience methods to maintain compatibility with existing code
    
    def get_market_price(self, symbol):
        """
        Get current market price for a symbol
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT')
            
        Returns:
            Current market price
        """
        # Convert BTCUSDT format to BTC/USD format
        if '/' not in symbol:
            base = symbol[:-4] if symbol.endswith('USDT') else symbol[:-3]
            quote = 'USDT' if symbol.endswith('USDT') else 'USD'
            formatted_symbol = f"{base}/{quote}"
        else:
            formatted_symbol = symbol
            
        ticker_data = self.get_market_ticker(formatted_symbol)
        
        if not ticker_data or not ticker_data.get("Success"):
            return None
            
        if "Data" in ticker_data and formatted_symbol in ticker_data["Data"]:
            return ticker_data["Data"][formatted_symbol].get("LastPrice")
            
        return None
    
    def get_historical_data(self, symbol, interval, limit=100):
        """
        Get historical market data
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT')
            interval: Candle interval (e.g., '15m')
            limit: Number of candles to fetch
            
        Returns:
            Historical market data
        """
        # This is a mock implementation since the API doesn't have historical data endpoint
        # In a real implementation, this would call the appropriate API endpoint
        
        # Get current price to use as a base for generating mock data
        current_price = self.get_market_price(symbol)
        
        if not isinstance(current_price, (int, float)):
            return {"error": "Could not get current price"}
            
        historical_data = []
        base_price = current_price * 0.95  # Start slightly below current price
        
        timestamp = int(time.time() * 1000)
        interval_ms = {
            "1m": 60 * 1000,
            "5m": 5 * 60 * 1000,
            "15m": 15 * 60 * 1000,
            "1h": 60 * 60 * 1000,
            "4h": 4 * 60 * 60 * 1000,
            "1d": 24 * 60 * 60 * 1000
        }.get(interval, 15 * 60 * 1000)  # Default to 15m
        
        for i in range(limit):
            open_price = base_price * (1 + (i / limit) * 0.1)
            high_price = open_price * 1.01
            low_price = open_price * 0.99
            close_price = open_price * (1 + (i % 3 - 1) * 0.005)  # Slight up/down pattern
            volume = 10000 + i * 100
            
            # Format: [timestamp, open, high, low, close, volume]
            candle = [timestamp - (limit - i) * interval_ms, 
                     open_price, high_price, low_price, close_price, volume]
            
            historical_data.append(candle)
            
        return historical_data
    
    def get_account_info(self):
        """
        Get account information (wrapper for get_account_balance)
        
        Returns:
            Dictionary with account information
        """
        return self.get_account_balance()
        
    def get_open_positions(self):
        """
        Get open positions (uses pending orders as a proxy)
        
        Returns:
            List of open positions
        """
        return self.query_order(pending_only=True)
        
    def get_trade_history(self, limit=50):
        """
        Get trade history
        
        Args:
            limit: Number of trades to fetch
            
        Returns:
            List of trades
        """
        return self.query_order(limit=limit)