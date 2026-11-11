/** Mount the station console.
 *
 * No StrictMode: it double-invokes effects in dev only, which would give
 * the mission event stream a different lifecycle in dev than in the build.
 */

import { createRoot } from "react-dom/client";

import { MissionConsole } from "@/app/MissionConsole";
import "@/app/console.css";

const root = document.getElementById("root");
if (root === null) throw new Error("index.html is missing #root");

createRoot(root).render(<MissionConsole />);
