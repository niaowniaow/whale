package main

import (
    "fmt"
    "math"
    "os"
    "strconv"
    "strings"
    "sync"
)

func playOne(a *Engine, b *Engine, goCmd string, maxPly int) (int, string) {
    history := ""
    turnWhite := true
    for ply := 0; ply < maxPly; ply++ {
        var mv string
        var err error
        if turnWhite {
            mv, err = a.goMoves(history, goCmd)
        } else {
            mv, err = b.goMoves(history, goCmd)
        }
        if err != nil || mv == "" {
            if turnWhite {
                return -1, history
            }
            return 1, history
        }
        if mv == "(none)" {
            if turnWhite {
                return -1, history
            }
            return 1, history
        }
        if history == "" {
            history = mv
        } else {
            history += " " + mv
        }
        turnWhite = !turnWhite
    }
    return 0, history
}

func eloDiff(score float64, total int) float64 {
    if total <= 0 {
        return 0
    }
    if score <= 0 {
        return -999
    }
    if score >= float64(total) {
        return 999
    }
    p := score / float64(total)
    return -400.0 * math.Log10(1.0/p-1.0)
}

func resultStr(whiteRes int) string {
    if whiteRes == 1 {
        return "1-0"
    }
    if whiteRes == -1 {
        return "0-1"
    }
    return "1/2-1/2"
}

func writePGN(f *os.File, mu *sync.Mutex, round int, whiteName string, blackName string, whiteRes int, moves string) {
    mu.Lock()
    defer mu.Unlock()
    fmt.Fprintf(f, "[Event \"whale-match\"]\n")
    fmt.Fprintf(f, "[Round \"%d\"]\n", round)
    fmt.Fprintf(f, "[White \"%s\"]\n", whiteName)
    fmt.Fprintf(f, "[Black \"%s\"]\n", blackName)
    fmt.Fprintf(f, "[Result \"%s\"]\n", resultStr(whiteRes))
    toks := strings.Fields(moves)
    var sb strings.Builder
    for i := 0; i < len(toks); i += 2 {
        sb.WriteString(strconv.Itoa(i/2 + 1))
        sb.WriteString(". ")
        sb.WriteString(toks[i])
        sb.WriteString(" ")
        if i+1 < len(toks) {
            sb.WriteString(toks[i+1])
            sb.WriteString(" ")
        }
    }
    sb.WriteString(resultStr(whiteRes))
    fmt.Fprintf(f, "%s\n\n", sb.String())
}
