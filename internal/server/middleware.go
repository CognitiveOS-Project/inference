package server

import (
	"encoding/json"
	"net/http"
	"sync"
	"time"
)

// rateLimiter implements a simple fixed-window rate limiter per client IP.
type rateLimiter struct {
	mu          sync.Mutex
	clients     map[string]*clientWindow
	window      time.Duration
	maxRequests int
}

type clientWindow struct {
	count      int
	windowStart time.Time
}

func newRateLimiter(window time.Duration, maxRequests int) *rateLimiter {
	return &rateLimiter{
		clients:     make(map[string]*clientWindow),
		window:      window,
		maxRequests: maxRequests,
	}
}

func (rl *rateLimiter) allow(ip string) bool {
	rl.mu.Lock()
	defer rl.mu.Unlock()

	now := time.Now()
	cw, ok := rl.clients[ip]
	if !ok || now.Sub(cw.windowStart) >= rl.window {
		rl.clients[ip] = &clientWindow{count: 1, windowStart: now}
		return true
	}

	if cw.count >= rl.maxRequests {
		return false
	}

	cw.count++
	return true
}

// validateJSONBody checks that the request body is valid JSON for POST requests.
func validateJSONBody(next http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodPost && r.Body != nil {
			var v interface{}
			if err := json.NewDecoder(r.Body).Decode(&v); err != nil {
				http.Error(w, `{"error":"invalid JSON body","code":"E_INVALID_JSON"}`, http.StatusBadRequest)
				return
			}
			_ = v
		}
		next.ServeHTTP(w, r)
	}
}

// rateLimitMiddleware wraps a handler with per-IP rate limiting.
func (s *Server) rateLimitMiddleware(next http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		ip := r.RemoteAddr
		if colon := len(r.RemoteAddr) - 1; colon >= 0 {
			for i := len(r.RemoteAddr) - 1; i >= 0; i-- {
				if r.RemoteAddr[i] == ':' {
					colon = i
					break
				}
			}
			ip = r.RemoteAddr[:colon]
		}

		if !s.rateLimiter.allow(ip) {
			http.Error(w, `{"error":"rate limit exceeded","code":"E_RATE_LIMITED"}`, http.StatusTooManyRequests)
			return
		}
		next.ServeHTTP(w, r)
	}
}

// middleware wraps handlers with validation and rate limiting.
func (s *Server) middleware(next http.HandlerFunc) http.HandlerFunc {
	return s.rateLimitMiddleware(validateJSONBody(next))
}
