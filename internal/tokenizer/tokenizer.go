// Package tokenizer mirrors training/tokenizer.py so Go inference produces
// byte-identical token IDs to the Python training code.
package tokenizer

import (
	"encoding/json"
	"fmt"
	"os"
	"regexp"
	"strings"

	"golang.org/x/text/unicode/norm"
)

const (
	PadID int64 = 0
	UnkID int64 = 1

	tokenURL   = "<url>"
	tokenPhone = "<phone>"
)

var (
	reURL   = regexp.MustCompile(`https?://\S+|www\.\S+`)
	rePhone = regexp.MustCompile(`\d{2,4}[-\s.]?\d{3,4}[-\s.]?\d{4}`)
	reWS    = regexp.MustCompile(`\s+`)
)

// Tokenizer is the Go counterpart of Python's CharTokenizer.
type Tokenizer struct {
	MaxLength int
	charToID  map[string]int64
}

type vocabFile struct {
	MaxLength int              `json:"max_length"`
	CharToID  map[string]int64 `json:"char_to_id"`
}

// Load reads a vocab.json produced by tokenizer.save() in Python.
func Load(path string) (*Tokenizer, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read vocab: %w", err)
	}
	var vf vocabFile
	if err := json.Unmarshal(raw, &vf); err != nil {
		return nil, fmt.Errorf("parse vocab: %w", err)
	}
	if vf.MaxLength <= 0 || len(vf.CharToID) == 0 {
		return nil, fmt.Errorf("vocab file missing required fields")
	}
	return &Tokenizer{MaxLength: vf.MaxLength, charToID: vf.CharToID}, nil
}

// Normalize matches Python's normalize(): NFKC, URL/phone placeholders,
// whitespace collapse, trim.
func Normalize(s string) string {
	s = norm.NFKC.String(s)
	s = reURL.ReplaceAllString(s, " "+tokenURL+" ")
	s = rePhone.ReplaceAllString(s, " "+tokenPhone+" ")
	s = reWS.ReplaceAllString(s, " ")
	return strings.TrimSpace(s)
}

// Encode converts text into a fixed-length slice of token IDs.
func (t *Tokenizer) Encode(text string) []int64 {
	ids := make([]int64, 0, t.MaxLength)
	normalized := Normalize(text)

	i := 0
	for i < len(normalized) && len(ids) < t.MaxLength {
		if matched := tryMatch(normalized, i, tokenURL); matched > 0 {
			ids = append(ids, t.lookup(tokenURL))
			i += matched
			continue
		}
		if matched := tryMatch(normalized, i, tokenPhone); matched > 0 {
			ids = append(ids, t.lookup(tokenPhone))
			i += matched
			continue
		}
		r, size := decodeRune(normalized[i:])
		ids = append(ids, t.lookup(string(r)))
		i += size
	}

	for len(ids) < t.MaxLength {
		ids = append(ids, PadID)
	}
	return ids
}

func (t *Tokenizer) lookup(tok string) int64 {
	if id, ok := t.charToID[tok]; ok {
		return id
	}
	return UnkID
}

func tryMatch(s string, i int, token string) int {
	if strings.HasPrefix(s[i:], token) {
		return len(token)
	}
	return 0
}

func decodeRune(s string) (rune, int) {
	for _, r := range s {
		return r, len(string(r))
	}
	return 0, 0
}
