package main

import (
    "bufio"
    "fmt"
    "os/exec"
    "strconv"
    "strings"
    "time"
)

type Engine struct {
    cmd *exec.Cmd
    in  *bufio.Writer
    out *bufio.Scanner
}

func startEngine(path string, hash int, threads int) (*Engine, error) {
    cmd := exec.Command(path)
    stdin, err := cmd.StdinPipe()
    if err != nil {
        return nil, err
    }
    stdout, err := cmd.StdoutPipe()
    if err != nil {
        return nil, err
    }
    if err := cmd.Start(); err != nil {
        return nil, err
    }
    e := &Engine{cmd: cmd, in: bufio.NewWriter(stdin), out: bufio.NewScanner(stdout)}
    e.out.Buffer(make([]byte, 65536), 65536)
    if err := e.send("uci"); err != nil {
        return nil, err
    }
    if err := e.waitFor("uciok", 10*time.Second); err != nil {
        return nil, err
    }
    if err := e.send("setoption name Hash value " + strconv.Itoa(hash)); err != nil {
        return nil, err
    }
    if err := e.send("setoption name Threads value " + strconv.Itoa(threads)); err != nil {
        return nil, err
    }
    if err := e.send("isready"); err != nil {
        return nil, err
    }
    if err := e.waitFor("readyok", 10*time.Second); err != nil {
        return nil, err
    }
    return e, nil
}

func (e *Engine) send(s string) error {
    if _, err := e.in.WriteString(s + "\n"); err != nil {
        return err
    }
    return e.in.Flush()
}

func (e *Engine) waitFor(token string, timeout time.Duration) error {
    deadline := time.Now().Add(timeout)
    for {
        if time.Now().After(deadline) {
            return fmt.Errorf("timeout waiting " + token)
        }
        if !e.out.Scan() {
            return fmt.Errorf("eof waiting " + token)
        }
        if strings.Contains(e.out.Text(), token) {
            return nil
        }
    }
}

func (e *Engine) goMoves(moves string, goCmd string) (string, error) {
    pos := "position startpos"
    if moves != "" {
        pos += " moves " + moves
    }
    if err := e.send(pos); err != nil {
        return "", err
    }
    if err := e.send(goCmd); err != nil {
        return "", err
    }
    deadline := time.Now().Add(300 * time.Second)
    for {
        if time.Now().After(deadline) {
            return "", fmt.Errorf("timeout bestmove")
        }
        if !e.out.Scan() {
            return "", fmt.Errorf("eof bestmove")
        }
        line := e.out.Text()
        if strings.HasPrefix(line, "bestmove") {
            f := strings.Fields(line)
            if len(f) >= 2 {
                return f[1], nil
            }
            return "", fmt.Errorf("bad bestmove")
        }
    }
}

func (e *Engine) quit() {
    _ = e.send("quit")
    _ = e.cmd.Wait()
}
