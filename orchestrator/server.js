import express from "express";
import { fileURLToPath } from "node:url";
import path from "node:path";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const app = express();
app.use(express.json());
app.use(express.static(path.join(__dirname, "public")));

app.get("/api/health", (_req, res) => res.json({ ok: true }));

const PORT = Number(process.env.PORT ?? 3100);
const HOST = process.env.HOST ?? "127.0.0.1";
app.listen(PORT, HOST, () => {
  console.log(`[orchestrator] listening on http://${HOST}:${PORT}`);
});
