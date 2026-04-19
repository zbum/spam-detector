package main

import (
	"context"
	"errors"
	"flag"
	"log"
	"net/http"
	"os"
	"os/signal"
	"runtime"
	"syscall"
	"time"

	"spam-detector/internal/detector"
	"spam-detector/internal/server"
)

func defaultORTLib() string {
	switch runtime.GOOS {
	case "darwin":
		if runtime.GOARCH == "arm64" {
			return "/opt/homebrew/lib/libonnxruntime.dylib"
		}
		return "/usr/local/lib/libonnxruntime.dylib"
	case "linux":
		return "/usr/lib/libonnxruntime.so"
	case "windows":
		return "onnxruntime.dll"
	default:
		return ""
	}
}

func main() {
	addr := flag.String("addr", ":8080", "HTTP listen address")
	onnxPath := flag.String("model", "training/artifacts/spam_detector.onnx", "path to ONNX model")
	vocabPath := flag.String("vocab", "training/artifacts/vocab.json", "path to vocab.json")
	ortLib := flag.String("ort-lib", defaultORTLib(), "path to onnxruntime shared library")
	flag.Parse()

	d, err := detector.New(*onnxPath, *vocabPath, *ortLib)
	if err != nil {
		log.Fatalf("init detector: %v", err)
	}
	defer d.Close()

	srv := &http.Server{
		Addr:              *addr,
		Handler:           server.New(d),
		ReadHeaderTimeout: 5 * time.Second,
		ReadTimeout:       15 * time.Second,
		WriteTimeout:      15 * time.Second,
		IdleTimeout:       60 * time.Second,
	}

	serveErr := make(chan error, 1)
	go func() {
		log.Printf("spam-detector listening on %s", *addr)
		if err := srv.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
			serveErr <- err
		}
	}()

	stop := make(chan os.Signal, 1)
	signal.Notify(stop, os.Interrupt, syscall.SIGTERM)

	select {
	case err := <-serveErr:
		log.Fatalf("server error: %v", err)
	case sig := <-stop:
		log.Printf("received %s, shutting down", sig)
	}

	shutdownCtx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	if err := srv.Shutdown(shutdownCtx); err != nil {
		log.Printf("shutdown error: %v", err)
	}
}
