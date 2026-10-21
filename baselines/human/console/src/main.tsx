/** Mount the capacity study's operator console (baselines/human/study_server.py serves it). */

import { createRoot } from "react-dom/client";

import { CapacityConsole } from "@/app/CapacityConsole";
import "@/app/console.css";

const root = document.getElementById("root");
if (root === null) throw new Error("index.html is missing #root");
createRoot(root).render(<CapacityConsole />);
