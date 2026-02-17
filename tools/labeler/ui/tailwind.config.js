/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        accent: {
          DEFAULT: "#00f3ff", // neon cyan
          hover: "#00d8e6",
        },
        border: "#333333",
        neon: {
          green: "#00ff41",
          red: "#ff003c",
          cyan: "#00f3ff",
          yellow: "#ffcc00",
        },
        background: {
          main: "#050505",
          panel: "#0d0d15",
        }
      },
      fontFamily: {
        cyber: ["Orbitron", "sans-serif"],
        rajdhani: ["Rajdhani", "sans-serif"],
      }
    },
  },
  plugins: [],
}
