// Package server exposes the spam detector over HTTP.
package server

import (
	"encoding/json"
	"errors"
	"log"
	"net/http"
	"time"

	"spam-detector/internal/detector"
)

const maxBatchSize = 128

// New wires the detector into an http.Handler with all routes registered.
func New(d *detector.Detector) http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /healthz", healthz)
	mux.HandleFunc("POST /classify", classifyHandler(d))
	mux.HandleFunc("POST /classify/batch", batchHandler(d))
	return logging(mux)
}

type classifyRequest struct {
	Text string `json:"text"`
}

type classifyResponse struct {
	IsSpam bool    `json:"isSpam"`
	PSpam  float32 `json:"pSpam"`
	PHam   float32 `json:"pHam"`
}

type batchRequest struct {
	Texts []string `json:"texts"`
}

type batchItem struct {
	Text   string  `json:"text"`
	IsSpam bool    `json:"isSpam"`
	PSpam  float32 `json:"pSpam"`
	PHam   float32 `json:"pHam"`
}

type batchResponse struct {
	Results []batchItem `json:"results"`
}

type errorResponse struct {
	Error string `json:"error"`
}

func healthz(w http.ResponseWriter, _ *http.Request) {
	writeJSON(w, http.StatusOK, map[string]string{"status": "ok"})
}

func classifyHandler(d *detector.Detector) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		var req classifyRequest
		if err := decodeJSON(r, &req); err != nil {
			writeError(w, http.StatusBadRequest, err)
			return
		}
		if req.Text == "" {
			writeError(w, http.StatusBadRequest, errors.New("'text' is required"))
			return
		}
		res, err := d.Classify(req.Text)
		if err != nil {
			writeError(w, http.StatusInternalServerError, err)
			return
		}
		writeJSON(w, http.StatusOK, classifyResponse{
			IsSpam: res.IsSpam,
			PSpam:  res.PSpam,
			PHam:   res.PHam,
		})
	}
}

func batchHandler(d *detector.Detector) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		var req batchRequest
		if err := decodeJSON(r, &req); err != nil {
			writeError(w, http.StatusBadRequest, err)
			return
		}
		if len(req.Texts) == 0 {
			writeError(w, http.StatusBadRequest, errors.New("'texts' must contain at least one message"))
			return
		}
		if len(req.Texts) > maxBatchSize {
			writeError(w, http.StatusBadRequest, errors.New("batch size exceeds limit"))
			return
		}

		items := make([]batchItem, 0, len(req.Texts))
		for _, t := range req.Texts {
			res, err := d.Classify(t)
			if err != nil {
				writeError(w, http.StatusInternalServerError, err)
				return
			}
			items = append(items, batchItem{
				Text:   t,
				IsSpam: res.IsSpam,
				PSpam:  res.PSpam,
				PHam:   res.PHam,
			})
		}
		writeJSON(w, http.StatusOK, batchResponse{Results: items})
	}
}

func decodeJSON(r *http.Request, dst any) error {
	dec := json.NewDecoder(r.Body)
	dec.DisallowUnknownFields()
	return dec.Decode(dst)
}

func writeJSON(w http.ResponseWriter, status int, body any) {
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(body)
}

func writeError(w http.ResponseWriter, status int, err error) {
	writeJSON(w, status, errorResponse{Error: err.Error()})
}

func logging(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		sw := &statusWriter{ResponseWriter: w, status: http.StatusOK}
		next.ServeHTTP(sw, r)
		log.Printf("%s %s %d %s", r.Method, r.URL.Path, sw.status, time.Since(start))
	})
}

type statusWriter struct {
	http.ResponseWriter
	status int
}

func (s *statusWriter) WriteHeader(code int) {
	s.status = code
	s.ResponseWriter.WriteHeader(code)
}
