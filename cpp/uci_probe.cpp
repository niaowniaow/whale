#include <cstdio>
#include <string>
int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: uci_probe <engine> <depth>\n");
        return 2;
    }
    std::string engine = argv[1];
    std::string depth = argv[2];
    std::string script = "uci\nisready\nposition startpos\ngo depth " + depth + "\n";
    std::string cmd = engine + " 2>/dev/null <<'EOF'\n" + script + "EOF\n";
    FILE* p = popen(cmd.c_str(), "r");
    if (!p) return 1;
    char buf[4096];
    std::string out;
    while (std::fgets(buf, sizeof(buf), p)) {
        out += buf;
        if (out.find("bestmove") != std::string::npos) break;
    }
    int rc = pclose(p);
    std::fputs(out.c_str(), stdout);
    return rc;
}
