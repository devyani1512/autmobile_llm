import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { ArrowLeft, Car, Moon, Sun } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";



interface Brand {
  id: string;
  name: string;
  tagline: string;
  theme: string;
  
  video: string; // should be full public path like "/videos/toyota.mp4"
  color?: string; // optional theme accent
}

const brands: Brand[] = [
  {
    id: "toyota",
    name: "TOYOTA",
    tagline: "Let's Go Places",
    theme: "silver",
   
    video: "/videos/toyota.mp4",
    color: "from-[#1f2937] to-[#111827]"
  },
  {
    id: "maruti",
    name: "MARUTI SUZUKI",
    tagline: "Way of Life",
    theme: "chrome",
   
    video: "/videos/suzuki.mp4",
    color: "from-[#0f172a] to-[#021124]"
  },
  {
    id: "hyundai",
    name: "HYUNDAI",
    tagline: "New Thinking. New Possibilities.",
    theme: "platinum",
   
    video: "/videos/hyundai.mp4",
    color: "from-[#04152b] to-[#081a2f]"
  },
  {
    id: "tata",
    name: "TATA MOTORS",
    tagline: "Connecting Aspirations",
    theme: "silver",
   
    video: "/videos/tata.mp4",
    color: "from-[#0b1220] to-[#07101a]"
  },
  {
    id: "nissan",
    name: "NISSAN",
    tagline: "Innovation That Excites",
    theme: "chrome",
    
    video: "/videos/nissan.mp4",
    color: "from-[#081026] to-[#04101b]"
  },
];

export default function BrandSelection() {
  const navigate = useNavigate();
  const [themeDark, setThemeDark] = useState(true);
  const [selected, setSelected] = useState<Brand | null>(null);

  const openBrand = (b: Brand) => setSelected(b);
  const closeBrand = () => setSelected(null);
  const goToModels = (brandId: string) => navigate(`/model-selection?brand=${brandId}`);

  return (
    
    <>
    <Stars />

    <div className={themeDark ? "min-h-screen bg-black text-white" : "min-h-screen bg-white text-slate-900"}>
      {/* Top bar */}
      <div className="border-b border-transparent/10 glass-card sticky top-0 z-30 backdrop-blur-md">
        <div className="container mx-auto px-4 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Button variant="ghost" onClick={() => navigate("/")} className="gap-2">
              <ArrowLeft className="w-4 h-4" />
              Back
            </Button>
            <div className="ml-2 text-sm text-muted-foreground/80 hidden sm:block">
              Home <span className="mx-2">›</span> Brand Selection
            </div>
          </div>

          <div className="flex items-center gap-4">
            <div className="text-xs uppercase tracking-widest text-muted-foreground/80 hidden md:block">
              Premium Brands
            </div>

            <button
              aria-label="Toggle theme"
              onClick={() => setThemeDark((s) => !s)}
              className="p-2 rounded-md hover:bg-white/5 transition"
            >
              {themeDark ? <Moon className="w-4 h-4" /> : <Sun className="w-4 h-4" />}
            </button>
          </div>
        </div>
      </div>

      {/* Hero */}
      <div className="container mx-auto px-4 py-12 md:py-20">
        <div className="text-center max-w-4xl mx-auto mb-10">
          <motion.h1 initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.6 }} className="text-4xl md:text-6xl font-extralight tracking-tight leading-tight">
            Automotive <span className="font-light">Intelligence</span>
          </motion.h1>
          <p className="mt-4 text-sm md:text-lg text-muted-foreground/80">
            SELECT YOUR BRAND
          </p>
        </div>

        {/* Grid */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-8">
          {brands.map((brand, idx) => (
            <motion.div
              key={brand.id}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.08 * idx, duration: 0.6 }}
              whileHover={{ y: -12 }}
              className="cursor-pointer"
            >
              <Card
             onClick={() => goToModels(brand.id)}

                className={`relative group overflow-hidden rounded-3xl h-72 md:h-80 p-6 border border-white/6 transition-all duration-500 transform ${themeDark ? "bg-gradient-to-br from-[#0b1220]/60 to-[#061026]/40" : "bg-white/5"}`}
              >
                {/* Hover video (fades in on hover) */}
                <div className="absolute inset-0 -z-10 overflow-hidden pointer-events-none">
                  <video
                    src={brand.video}
                    autoPlay
                    loop
                    muted
                    playsInline
                    /*className="w-full h-full object-cover opacity-0 group-hover:opacity-70 transition-opacity duration-700 scale-105 group-hover:scale-100"*/
                    className="w-full h-full object-cover opacity-40 group-hover:opacity-70 transition-opacity duration-700"

                  />
                </div>

                {/* Glass overlay + subtle gradient */}
                <div className="absolute inset-0 bg-gradient-to-b from-black/40 via-transparent to-black/40 opacity-60 pointer-events-none" />

                {/* Accent stroke */}
                <div className="absolute -top-8 -left-16 w-48 h-48 rounded-full bg-gradient-to-br from-white/3 to-transparent opacity-5 pointer-events-none blur-2xl" />

                {/* Foreground content */}
                <div className="relative z-10 flex flex-col items-center justify-center h-full text-center px-2">
  <h3 className="text-2xl md:text-3xl font-semibold tracking-tight opacity-100 group-hover:opacity-0 transition duration-300">{brand.name}</h3>

<p className="text-sm text-muted-foreground/70 italic mt-2 mb-3 opacity-100 group-hover:opacity-0 transition duration-300">{brand.tagline}</p>


  <div className="mt-3 opacity-0 group-hover:opacity-100 transition-opacity duration-500">
    
  </div>
</div>

              </Card>
            </motion.div>
          ))}
        </div>
      </div>

      {/* Expand modal / preview */}
      <AnimatePresence>
        {selected && (
          <motion.div
            className="fixed inset-0 z-50 flex items-center justify-center p-6"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
          >
            <div className="absolute inset-0 bg-black/70 backdrop-blur-sm" onClick={closeBrand} />

            <motion.div
              initial={{ scale: 0.98, y: 20, opacity: 0 }}
              animate={{ scale: 1, y: 0, opacity: 1 }}
              exit={{ scale: 0.98, y: 20, opacity: 0 }}
              transition={{ duration: 0.35 }}
              className="relative max-w-4xl w-full bg-gradient-to-br from-[#081026] to-[#04101b] rounded-2xl overflow-hidden border border-white/8 shadow-2xl"
            >
              {/* big video */}
              <div className="relative h-72 md:h-96">
                <video
                  src={selected.video}
                  autoPlay
                  controls
                  muted={false}
                  playsInline
                  className="w-full h-full object-cover"
                />
                {/* top actions */}
                <div className="absolute top-4 left-4 flex gap-2">
                  <button onClick={closeBrand} className="p-2 rounded-md bg-black/40 hover:bg-black/30">
                    Close
                  </button>
                </div>
              </div>

              <div className="p-6 md:p-8 flex flex-col md:flex-row md:items-center md:justify-between gap-4">
                <div>
                  <h2 className="text-2xl md:text-3xl font-semibold">{selected.name}</h2>
                  <p className="mt-1 text-sm text-muted-foreground/80 italic">{selected.tagline}</p>
                  <p className="mt-3 text-sm text-muted-foreground/70 max-w-xl">
                    Cinematic preview. Click continue to view available models and specs.
                  </p>
                </div>

                <div className="flex items-center gap-3">
                  <Button onClick={() => goToModels(selected.id)} className="px-6 py-3">
                    Continue
                  </Button>

                  <Button variant="ghost" onClick={closeBrand}>
                    Close
                  </Button>
                </div>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
   </>
  );
}
const Stars = () => {
  return (
    <div
      style={{
        position: "fixed",
        inset: 0,
        zIndex: -1,
        background:
          "radial-gradient(circle at 50% 50%, rgba(255,255,255,0.5) 1px, transparent 1px)",
        backgroundSize: "3px 3px",
        opacity: 0.35,
      }}
    ></div>
  );
};
