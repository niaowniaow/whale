package main

import (
    "fmt"
    "os"
    "strconv"
    "sync"
)

func main() {
    if len(os.Args) < 5 {
        fmt.Fprintln(os.Stderr, "usage: whale-match <engA> <engB> <games> <depth> [workers] [pgn] [hash] [threads]")
        os.Exit(2)
    }
    engA := os.Args[1]
    engB := os.Args[2]
    games, _ := strconv.Atoi(os.Args[3])
    depth, _ := strconv.Atoi(os.Args[4])
    workers := 1
    if len(os.Args) >= 6 {
        workers, _ = strconv.Atoi(os.Args[5])
    }
    pgnPath := ""
    if len(os.Args) >= 7 {
        pgnPath = os.Args[6]
    }
    hash := 16
    if len(os.Args) >= 8 {
        hash, _ = strconv.Atoi(os.Args[7])
    }
    threads := 1
    if len(os.Args) >= 9 {
        threads, _ = strconv.Atoi(os.Args[8])
    }
    if games <= 0 {
        games = 2
    }
    if depth <= 0 {
        depth = 8
    }
    if workers <= 0 {
        workers = 1
    }
    goCmd := "go depth " + strconv.Itoa(depth)
    var pgnFile *os.File
    var pgnMu sync.Mutex
    if pgnPath != "" {
        var err error
        pgnFile, err = os.Create(pgnPath)
        if err != nil {
            fmt.Fprintln(os.Stderr, err)
            os.Exit(1)
        }
        defer pgnFile.Close()
    }
    var mu sync.Mutex
    scoreA := 0.0
    wins := 0
    draws := 0
    losses := 0
    jobs := make(chan int, games)
    for g := 0; g < games; g++ {
        jobs <- g
    }
    close(jobs)
    var wg sync.WaitGroup
    for w := 0; w < workers; w++ {
        wg.Add(1)
        go func() {
            defer wg.Done()
            a, err := startEngine(engA, hash, threads)
            if err != nil {
                fmt.Fprintln(os.Stderr, err)
                return
            }
            defer a.quit()
            b, err := startEngine(engB, hash, threads)
            if err != nil {
                fmt.Fprintln(os.Stderr, err)
                return
            }
            defer b.quit()
            for g := range jobs {
                var whiteRes int
                var hist string
                var resA int
                var wName, bName string
                if g%2 == 0 {
                    r, h := playOne(a, b, goCmd, 200)
                    whiteRes, hist = r, h
                    resA = r
                    wName, bName = "A", "B"
                } else {
                    r, h := playOne(b, a, goCmd, 200)
                    whiteRes, hist = r, h
                    resA = -r
                    wName, bName = "B", "A"
                }
                mu.Lock()
                if resA == 1 {
                    wins++
                    scoreA++
                } else if resA == -1 {
                    losses++
                } else {
                    draws++
                    scoreA += 0.5
                }
                done := wins + draws + losses
                fmt.Printf("game %d/%d white=%s res=%s scoreA=%.1f\n", done, games, wName+bName, resultStr(whiteRes), scoreA)
                mu.Unlock()
                if pgnFile != nil {
                    writePGN(pgnFile, &pgnMu, g+1, wName, bName, whiteRes, hist)
                }
            }
        }()
    }
    wg.Wait()
    fmt.Printf("final A:+%d=%d-%d %.1f/%d elo=%+.1f\n", wins, draws, losses, scoreA, games, eloDiff(scoreA, games))
}
