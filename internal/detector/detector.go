// Package detector runs the exported Char-CNN ONNX model for spam classification.
package detector

import (
	"fmt"
	"math"
	"sync"

	ort "github.com/yalue/onnxruntime_go"

	"spam-detector/internal/tokenizer"
)

// Result is a single classification outcome.
type Result struct {
	IsSpam  bool
	PSpam   float32
	PHam    float32
}

// Detector wraps a loaded ONNX session plus its tokenizer.
// The shared session + tensors are not concurrency-safe, so Classify
// serializes access with a mutex. For higher throughput, run multiple
// Detector instances (each with its own session).
type Detector struct {
	tok     *tokenizer.Tokenizer
	session *ort.AdvancedSession
	input   *ort.Tensor[int64]
	output  *ort.Tensor[float32]
	mu      sync.Mutex
}

// New loads vocab.json and the ONNX model. Caller must Close when done.
// ortLibPath points to the platform-specific ONNX Runtime shared lib.
func New(onnxPath, vocabPath, ortLibPath string) (*Detector, error) {
	tok, err := tokenizer.Load(vocabPath)
	if err != nil {
		return nil, err
	}

	if ortLibPath != "" {
		ort.SetSharedLibraryPath(ortLibPath)
	}
	if err := ort.InitializeEnvironment(); err != nil {
		return nil, fmt.Errorf("init onnxruntime: %w", err)
	}

	inputShape := ort.NewShape(1, int64(tok.MaxLength))
	inputTensor, err := ort.NewEmptyTensor[int64](inputShape)
	if err != nil {
		return nil, fmt.Errorf("alloc input tensor: %w", err)
	}

	outputShape := ort.NewShape(1, 2)
	outputTensor, err := ort.NewEmptyTensor[float32](outputShape)
	if err != nil {
		return nil, fmt.Errorf("alloc output tensor: %w", err)
	}

	sess, err := ort.NewAdvancedSession(
		onnxPath,
		[]string{"input_ids"},
		[]string{"logits"},
		[]ort.Value{inputTensor},
		[]ort.Value{outputTensor},
		nil,
	)
	if err != nil {
		return nil, fmt.Errorf("create session: %w", err)
	}

	return &Detector{
		tok:     tok,
		session: sess,
		input:   inputTensor,
		output:  outputTensor,
	}, nil
}

// Classify runs inference on a single message. Safe for concurrent use.
func (d *Detector) Classify(text string) (Result, error) {
	ids := d.tok.Encode(text)

	d.mu.Lock()
	defer d.mu.Unlock()

	buf := d.input.GetData()
	if len(buf) != len(ids) {
		return Result{}, fmt.Errorf("tensor/ids length mismatch: %d vs %d", len(buf), len(ids))
	}
	copy(buf, ids)

	if err := d.session.Run(); err != nil {
		return Result{}, fmt.Errorf("run session: %w", err)
	}

	logits := d.output.GetData()
	if len(logits) != 2 {
		return Result{}, fmt.Errorf("unexpected output length: %d", len(logits))
	}
	pHam, pSpam := softmax2(logits[0], logits[1])
	return Result{
		IsSpam: pSpam > pHam,
		PSpam:  pSpam,
		PHam:   pHam,
	}, nil
}

// Close releases ONNX Runtime resources.
func (d *Detector) Close() error {
	if d.session != nil {
		if err := d.session.Destroy(); err != nil {
			return err
		}
	}
	if d.input != nil {
		_ = d.input.Destroy()
	}
	if d.output != nil {
		_ = d.output.Destroy()
	}
	return ort.DestroyEnvironment()
}

func softmax2(a, b float32) (float32, float32) {
	m := a
	if b > m {
		m = b
	}
	ea := float32(math.Exp(float64(a - m)))
	eb := float32(math.Exp(float64(b - m)))
	s := ea + eb
	return ea / s, eb / s
}
