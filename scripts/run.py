"""Megazord: le o megazord.ini e roda o modo escolhido. Chamado pelo run.bat.

MODO = treino   -> os traders do time continuam estudando, rodada apos rodada (Ctrl+C para parar)
MODO = replay   -> os campeoes jogam os ultimos pregoes do historico, para assistir
MODO = ao_vivo  -> os campeoes votam a cada candle novo (MT5 ou pregao simulado). So painel, sem ordens.
"""
import functools
import http.server
import json
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from winrl.data import load, split  # noqa: E402
from winrl.env import WinDayTradeEnv  # noqa: E402
from winrl.live import Board  # noqa: E402
from winrl.team import splits, load_settings, play_day, play_days, probs_fn, seed_team, team  # noqa: E402

PY = sys.executable
INI = str(ROOT / "megazord.ini")


# ----------------------------------------------------------------------------- painel
def data_dir(s):
    """Pasta dos dados do painel: dashboard/data para o time principal, dashboard/data_<pasta> para os outros
    (assim um teste nunca sobrescreve o painel do time de verdade)."""
    return ROOT / "dashboard" / ("data" if s.pasta_time == "team" else f"data_{s.pasta_time}")


def start_dashboard(port: int, open_browser=True, page="escritorio", data=None):
    data = data or ROOT / "dashboard" / "data"

    class Quiet(http.server.SimpleHTTPRequestHandler):
        def translate_path(self, path):
            clean = path.split("?", 1)[0].split("#", 1)[0]
            if clean.startswith("/data/"):
                return str(data / clean[len("/data/"):])
            return super().translate_path(path)

        def end_headers(self):
            self.send_header("Cache-Control", "no-store")
            super().end_headers()

        def log_message(self, *a):
            pass

    try:
        srv = http.server.ThreadingHTTPServer(("127.0.0.1", port), functools.partial(Quiet, directory=str(ROOT / "dashboard")))
    except OSError:
        print(f"(painel ja estava rodando na porta {port})")
    else:
        threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://localhost:{port}/{'time' if page == 'painel' else 'escritorio'}.html"
    print(f"Escritorio: http://localhost:{port}/escritorio.html")
    print(f"Painel:     http://localhost:{port}/time.html")
    if open_browser:
        webbrowser.open(url)


# ----------------------------------------------------------------------------- time
def ensure_team(s, need_champions: bool):
    trs = team(s)
    missing = [t for t in trs if not t.learner.exists()]
    if missing:
        # aproveita os modelos ja treinados com 5 anos de dados, se existirem
        srcs = [ROOT / "models" / f"pen5y_WIN_N_s{t.idx}" / "best_model.zip" for t in trs]
        if all(p.exists() for p in srcs):
            for m in seed_team(s, srcs):
                print("time criado:", m)
        elif need_champions:
            raise SystemExit("O time ainda nao existe. Rode primeiro com MODO = treino.")
        else:
            print("time novo: os traders vao comecar do zero")
    return trs


def champions(s):
    trs = ensure_team(s, need_champions=True)
    from stable_baselines3 import PPO
    fns = [probs_fn(PPO.load(t.champion, device="cpu")) for t in trs]
    return trs, fns


def team_card(trs):
    """Ficha de cada trader para o painel (replay / ao vivo)."""
    rows = []
    for t in trs:
        h = t.history
        champ = next((r for r in reversed(h) if r["promoted"]), None)
        rows.append({"name": t.name, "rounds": len(h), "promotions": sum(r["promoted"] for r in h),
                     "steps": h[-1]["total_timesteps"] if h else None,
                     "val_pts_dia": champ["learner"]["pts_dia"] if champ else None})
    return rows


def team_history(s):
    f = s.team_dir / "team.json"
    return json.loads(f.read_text()) if f.exists() else []


def update_data(s):
    print("atualizando dados do MT5...")
    r = subprocess.run([PY, "scripts/export_rates.py", s.simbolo, "M1"], cwd=ROOT, capture_output=True, text=True)
    print(r.stdout.strip() or r.stderr.strip()[-300:] or "(sem saida)")


# ----------------------------------------------------------------------------- modos
def mode_treino(s):
    trs = ensure_team(s, need_champions=False)
    board = Board("treino", "Megazord treinando", {**s.env.to_dict(), "traders": s.traders}, data_dir=data_dir(s))
    hist = team_history(s)
    rnd = (hist[-1]["round"] + 1) if hist else 1
    done_rounds = 0
    while True:
        if s.atualizar_dados:
            update_data(s)
        print(f"\n=== rodada {rnd}: {len(trs)} traders estudando {s.passos_por_rodada:,} passos cada ===")
        procs = []
        for t in trs:
            t.dir.mkdir(parents=True, exist_ok=True)
            (t.dir / "progress.json").unlink(missing_ok=True)
            log = open(ROOT / "logs" / f"team_r{rnd}_{t.name}.log", "w")
            procs.append(subprocess.Popen([PY, "-W", "ignore", "scripts/team_round.py", "--trader", str(t.idx),
                                           "--round", str(rnd), "--ini", INI], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT))
        try:
            while any(p.poll() is None for p in procs):
                publish_progress(board, trs, rnd, s, "estudando")
                time.sleep(2)
        except KeyboardInterrupt:
            for p in procs:
                p.terminate()
            print("\nparado. O que cada trader ja tinha aprendido nas rodadas anteriores esta salvo em team/.")
            return
        failed = [t.name for t, p in zip(trs, procs) if p.returncode != 0]
        if failed:
            raise SystemExit(f"falhou: {failed}. Veja logs/team_r{rnd}_<nome>.log")
        publish_progress(board, trs, rnd, s, "prova do time")

        # prova do time: campeoes + Megazord nos ultimos pregoes de teste (nunca estudados)
        _, fns = champions(s)
        _, _, test = splits(s)
        base = WinDayTradeEnv(test, s.env, random_days=False)
        days = list(range(max(0, len(base.days) - s.dias_teste_painel), len(base.days)))
        mz, ind = play_days(base, fns, [t.name for t in trs], s, days)
        ms = mz.summary("MEGAZORD")
        row = {"round": rnd, "time": time.strftime("%Y-%m-%d %H:%M"), "test_days": len(days),
               "megazord": ms, "traders": [{"name": t.name, **r.summary(t.name),
                                            "promoted": t.history[-1]["promoted"] if t.history else None}
                                           for t, r in zip(trs, ind)]}
        (s.team_dir / "team.json").write_text(json.dumps(hist + [row], indent=1, default=str))
        hist.append(row)
        print(f"rodada {rnd}: MEGAZORD {ms.get('total pts', 0):+} pts em {len(days)} pregoes ({ms.get('trades', 0)} trades)")
        for t, r in zip(trs, ind):
            h = t.history[-1]
            print(f"  {t.name:8} {'PROMOVIDO' if h['promoted'] else 'mantido  '} validacao {h['learner']['pts_dia']:+.1f} pts/dia"
                  f" | teste {r.summary().get('total pts', 0):+} pts")
        ep, _ = play_day(base, fns, [t.name for t in trs], s, days[(rnd * 7) % len(days)])
        board.add_episode(ep, round=rnd)
        publish_progress(board, trs, rnd, s, "rodada concluida",
                         last_eval={"days": len(days), "mz_pts": ms.get("total pts", 0), "mz_trades": ms.get("trades", 0)})
        rnd += 1
        done_rounds += 1
        if s.rodadas and done_rounds >= s.rodadas:
            print("numero de rodadas do megazord.ini atingido")
            return


def history_rows(s):
    """Resultado da prova do time (teste: pregoes nunca usados) em cada rodada, para os graficos."""
    rows = []
    for r in team_history(s):
        m = r["megazord"]
        rows.append({"round": r["round"], "time": r["time"][11:16], "days": r["test_days"],
                     "mz_pts": m.get("total pts", 0), "mz_trades": m.get("trades", 0), "mz_hit": m.get("acerto"),
                     "promoted": sum(1 for t in r["traders"] if t.get("promoted")),
                     "traders": [t.get("total pts", 0) for t in r["traders"]]})
    return rows[-40:]


def publish_progress(board, trs, rnd, s, phase, **extra):
    rows = []
    for t in trs:
        pr = t.progress()
        champ = (pr.get("champion") or {}) if pr else {}
        rows.append({"name": t.name, "phase": pr.get("phase", "na fila"), "steps_done": pr.get("steps_done", 0),
                     "steps_total": pr.get("steps_total", s.passos_por_rodada), "promoted": pr.get("promoted"),
                     "champion_pts_dia": champ.get("pts_dia")})
    prog = sum(min(1, r["steps_done"] / max(r["steps_total"], 1)) for r in rows) / max(len(rows), 1)
    prev = board.index.get("team") or {}
    board.set_team({"round": rnd, "phase": phase, "round_progress": prog, "traders": rows,
                    "last_eval": extra.get("last_eval", prev.get("last_eval")), "history": history_rows(s)})


def mode_replay(s):
    trs, fns = champions(s)
    _, _, test = splits(s)
    base = WinDayTradeEnv(test, s.env, random_days=False)
    board = Board("replay", "Megazord replay", {**s.env.to_dict()}, keep=10_000, data_dir=data_dir(s))
    board.set_team({"static": team_card(trs), "history": history_rows(s)})
    days = range(max(0, len(base.days) - s.dias_teste_painel), len(base.days))
    print(f"os campeoes vao jogar {len(days)} pregoes de teste (nunca estudados)...")
    for k in days:
        ep, _ = play_day(base, fns, [t.name for t in trs], s, k)
        board.add_episode(ep)
        print(f"  {ep['day']}: MEGAZORD {ep['summary']['trades']} trades {ep['summary']['pnl_pts']:+.0f} pts | "
              + " ".join(f"{x['name']} {x['pnl_pts']:+.0f}" for x in ep['summary']['traders']))
    print("pronto. O painel continua aberto (Ctrl+C para sair).")
    while True:
        time.sleep(3600)


def mode_ao_vivo(s):
    from winrl.live_team import MT5Feed, SimFeed, run_live
    broker = None
    if s.enviar_ordens:
        if not s.conectar_mt5:
            raise SystemExit("ENVIAR_ORDENS = true precisa de CONECTAR_MT5 = true.")
        from winrl.broker import MT5Broker
        broker = MT5Broker(s)   # conta real: so com PERMITIR_CONTA_REAL e confirmacao digitada
    trs, fns = champions(s)
    if s.conectar_mt5:
        feed = MT5Feed(s.simbolo, allow_real=s.permitir_conta_real)
    else:
        df = load(s.simbolo)
        feed = SimFeed(df, s.velocidade_simulada)
        # no simulado, o fechamento e o do proprio pregao reproduzido
        s.fechamento = df[df["time"].dt.date == feed.day]["time"].max().strftime("%H:%M")
    board = Board("ao_vivo", "Megazord ao vivo", {**s.env.to_dict()}, data_dir=data_dir(s))
    board.set_team({"static": team_card(trs), "history": history_rows(s)})
    board.live_status(feed.label, "conectando...")
    try:
        run_live(feed, fns, [t.name for t in trs], s, board, broker=broker)
    except KeyboardInterrupt:
        if broker and broker.position() != 0:
            print("\nzerando a posicao antes de sair...")
            broker.flatten("encerrado")
        print("\nparado.")
        return
    print("O painel continua aberto (Ctrl+C para sair).")
    while True:
        time.sleep(3600)


if __name__ == "__main__":
    if len(sys.argv) > 1:  # outro arquivo de configuracao (ex.: testes)
        INI = str(Path(sys.argv[1]).resolve())
    s = load_settings(Path(INI))
    (ROOT / "logs").mkdir(exist_ok=True)
    print(f"MEGAZORD | modo {s.modo} | {s.traders} traders | entra com {s.votos_para_entrar}, mantem com {s.votos_para_manter}"
          f" | CONECTAR_MT5={s.conectar_mt5} ENVIAR_ORDENS={s.enviar_ordens}")
    start_dashboard(s.porta, s.abrir_navegador, s.pagina, data_dir(s))
    try:
        {"treino": mode_treino, "replay": mode_replay, "ao_vivo": mode_ao_vivo}[s.modo](s)
    except KeyboardInterrupt:
        print("\nparado.")
