import { useNavigate, useSearchParams } from "react-router-dom";
import { ArrowLeft, Car } from "lucide-react";
import { motion } from "framer-motion";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";

interface CarModel {
  id: string;
  name: string;
  tagline: string;
  specs: {
    engine: string;
    power: string;
    year: string;
  };
}

const modelsByBrand: Record<string, CarModel[]> = {
  mercedes: [
    { id: "s-class", name: "S-Class", tagline: "The Pinnacle of Luxury", specs: { engine: "V8 Twin-Turbo", power: "496 HP", year: "2024" } },
    { id: "e-class", name: "E-Class", tagline: "Business Elegance", specs: { engine: "Inline-6 Turbo", power: "255 HP", year: "2024" } },
    { id: "gle", name: "GLE SUV", tagline: "Luxury Utility", specs: { engine: "Inline-6 Turbo", power: "375 HP", year: "2024" } },
  ],
  bmw: [
    { id: "7-series", name: "7 Series", tagline: "Executive Flagship", specs: { engine: "V8 Twin-Turbo", power: "523 HP", year: "2024" } },
    { id: "5-series", name: "5 Series", tagline: "Dynamic Luxury", specs: { engine: "Inline-6 Turbo", power: "335 HP", year: "2024" } },
    { id: "x7", name: "X7 SUV", tagline: "Premium Family", specs: { engine: "Inline-6 Turbo", power: "335 HP", year: "2024" } },
  ],
  audi: [
    { id: "a8", name: "A8", tagline: "Sophisticated Excellence", specs: { engine: "V8 TFSI", power: "453 HP", year: "2024" } },
    { id: "a6", name: "A6", tagline: "Progressive Performance", specs: { engine: "V6 TFSI", power: "335 HP", year: "2024" } },
    { id: "q7", name: "Q7", tagline: "Elegant Presence", specs: { engine: "V6 TFSI", power: "335 HP", year: "2024" } },
  ],
  porsche: [
    { id: "taycan", name: "Taycan", tagline: "Soul, Electrified", specs: { engine: "Electric Dual Motor", power: "670 HP", year: "2024" } },
    { id: "panamera", name: "Panamera", tagline: "Four-Door Sports Car", specs: { engine: "V8 Twin-Turbo", power: "620 HP", year: "2024" } },
    { id: "cayenne", name: "Cayenne", tagline: "Sports Car DNA", specs: { engine: "V8 Twin-Turbo", power: "541 HP", year: "2024" } },
  ],
  lexus: [
    { id: "ls", name: "LS", tagline: "Flagship Luxury", specs: { engine: "V6 Twin-Turbo", power: "416 HP", year: "2024" } },
    { id: "es", name: "ES", tagline: "Refined Elegance", specs: { engine: "V6", power: "302 HP", year: "2024" } },
    { id: "rx", name: "RX", tagline: "Luxury Redefined", specs: { engine: "V6", power: "275 HP", year: "2024" } },
  ],
  tesla: [
    { id: "model-s", name: "Model S", tagline: "Electric Luxury", specs: { engine: "Electric Tri-Motor", power: "1020 HP", year: "2024" } },
    { id: "model-x", name: "Model X", tagline: "Falcon Wing Doors", specs: { engine: "Electric Tri-Motor", power: "1020 HP", year: "2024" } },
    { id: "model-3", name: "Model 3", tagline: "Performance Sedan", specs: { engine: "Electric Dual Motor", power: "480 HP", year: "2024" } },
  ],
  jaguar: [
    { id: "xj", name: "XJ", tagline: "British Luxury", specs: { engine: "V8 Supercharged", power: "550 HP", year: "2024" } },
    { id: "xf", name: "XF", tagline: "Sporting Grace", specs: { engine: "V6 Supercharged", power: "380 HP", year: "2024" } },
    { id: "f-pace", name: "F-PACE", tagline: "Performance SUV", specs: { engine: "V8 Supercharged", power: "550 HP", year: "2024" } },
  ],
  volvo: [
    { id: "s90", name: "S90", tagline: "Swedish Serenity", specs: { engine: "Inline-4 Turbo", power: "316 HP", year: "2024" } },
    { id: "xc90", name: "XC90", tagline: "Scandinavian SUV", specs: { engine: "Inline-4 Turbo", power: "316 HP", year: "2024" } },
    { id: "xc60", name: "XC60", tagline: "Compact Luxury", specs: { engine: "Inline-4 Turbo", power: "295 HP", year: "2024" } },
  ],
  genesis: [
    { id: "g90", name: "G90", tagline: "Modern Luxury", specs: { engine: "V6 Twin-Turbo", power: "365 HP", year: "2024" } },
    { id: "g80", name: "G80", tagline: "Athletic Elegance", specs: { engine: "V6 Twin-Turbo", power: "300 HP", year: "2024" } },
    { id: "gv80", name: "GV80", tagline: "Elevated Design", specs: { engine: "V6 Twin-Turbo", power: "375 HP", year: "2024" } },
  ],
};

const brandNames: Record<string, string> = {
  mercedes: "Mercedes-Benz",
  bmw: "BMW",
  audi: "Audi",
  porsche: "Porsche",
  lexus: "Lexus",
  tesla: "Tesla",
  jaguar: "Jaguar",
  volvo: "Volvo",
  genesis: "Genesis",
};

const ModelSelection = () => {
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const brandId = searchParams.get("brand") || "mercedes";
  const brandName = brandNames[brandId];
  const models = modelsByBrand[brandId] || modelsByBrand.mercedes;

  const cardVariants = {
    hidden: { opacity: 0, y: 40 },
    visible: (i: number) => ({
      opacity: 1,
      y: 0,
      transition: {
        delay: i * 0.15,
        duration: 0.7,
        ease: [0.16, 1, 0.3, 1] as const
      }
    })
  };

  const handleModelSelect = (modelId: string) => {
    navigate(`/mode-selection?brand=${brandId}&model=${modelId}`);
  };

  return (
    <div className="min-h-screen bg-gradient-hero relative overflow-hidden">
      {/* Animated background */}
      <div className="absolute inset-0 opacity-20">
        <div className="absolute top-1/3 left-1/3 w-[600px] h-[600px] bg-chrome/30 rounded-full blur-3xl animate-float" />
        <div className="absolute bottom-1/4 right-1/3 w-[500px] h-[500px] bg-secondary/20 rounded-full blur-3xl animate-float" style={{ animationDelay: "1.5s" }} />
      </div>

      {/* Header */}
      <header className="border-b border-border glass-card relative z-10">
        <div className="container mx-auto px-4 py-6">
          <div className="flex items-center justify-between">
            <Button
              variant="ghost"
              onClick={() => navigate("/brand-selection")}
              className="gap-2 hover:bg-secondary/10"
            >
              <ArrowLeft className="w-4 h-4" />
              Change Manufacturer
            </Button>
            <div className="text-sm text-muted-foreground">
              Home <span className="text-foreground mx-2">›</span> {brandName} <span className="text-foreground mx-2">›</span> Model Selection
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <div className="container mx-auto px-4 py-16 relative z-10">
        {/* Title Section */}
        <motion.div
          initial={{ opacity: 0, y: -30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
          className="text-center mb-16"
        >
          <div className="inline-flex items-center gap-3 px-6 py-3 rounded-full glass-card mb-6 border border-secondary/30">
            <Car className="w-5 h-5 text-secondary" />
            <span className="text-sm font-medium text-gradient-silver">{brandName}</span>
          </div>
          
          <h1 className="text-5xl md:text-7xl font-bold mb-6 leading-tight">
            <span className="text-foreground">
              Select Your
            </span>
            <br />
            <span className="text-gradient-chrome">
              {brandName} Model
            </span>
          </h1>
          
          <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
            Choose from our premium collection of {brandName} vehicles
          </p>
        </motion.div>

        {/* Model Cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-10 max-w-7xl mx-auto">
          {models.map((model, index) => (
            <motion.div
              key={model.id}
              custom={index}
              variants={cardVariants}
              initial="hidden"
              animate="visible"
              whileHover={{ 
                y: -20,
                scale: 1.03,
                transition: { duration: 0.5, ease: [0.4, 0, 0.2, 1] as const }
              }}
              onClick={() => handleModelSelect(model.id)}
              className="cursor-pointer"
            >
              <Card className="relative overflow-hidden glass-card border-2 border-transparent hover:border-chrome/50 transition-all duration-500 group">
                {/* Car Image Section - Top 60% */}
                <div className="relative h-64 bg-muted/20 overflow-hidden">
                  {/* Placeholder for car image */}
                  <div className="absolute inset-0 flex items-center justify-center bg-gradient-to-br from-muted/30 to-background group-hover:scale-110 transition-transform duration-700">
                    <Car className="w-32 h-32 text-secondary/30" />
                  </div>
                  
                  {/* Reflection effect */}
                  <div className="absolute inset-0 bg-gradient-to-t from-transparent via-transparent to-foreground/5 opacity-0 group-hover:opacity-100 transition-opacity duration-500" />
                </div>

                {/* Info Section - Bottom 40% */}
                <div className="p-8 relative">
                  <div className="absolute inset-0 bg-gradient-chrome opacity-0 group-hover:opacity-10 transition-opacity duration-500" />
                  <div className="absolute inset-0 shadow-glow-chrome opacity-0 group-hover:opacity-100 transition-opacity duration-500" />
                  
                  <div className="relative z-10">
                    {/* Model Name */}
                    <h3 className="text-3xl font-bold mb-2 text-gradient-chrome">
                      {model.name}
                    </h3>
                    
                    {/* Tagline */}
                    <p className="text-muted-foreground italic mb-6 text-sm">
                      {model.tagline}
                    </p>
                    
                    {/* Specs */}
                    <div className="space-y-2 mb-6 text-sm">
                      <div className="flex items-center gap-2">
                        <div className="w-1.5 h-1.5 rounded-full bg-secondary" />
                        <span className="text-muted-foreground">{model.specs.engine}</span>
                      </div>
                      <div className="flex items-center gap-2">
                        <div className="w-1.5 h-1.5 rounded-full bg-secondary" />
                        <span className="text-muted-foreground">{model.specs.power}</span>
                      </div>
                      <div className="flex items-center gap-2">
                        <div className="w-1.5 h-1.5 rounded-full bg-secondary" />
                        <span className="text-muted-foreground">{model.specs.year}</span>
                      </div>
                    </div>
                    
                    {/* Button */}
                    <Button className="w-full bg-gradient-chrome hover:opacity-90 text-background font-semibold py-6 group-hover:scale-105 transition-transform duration-300">
                      Select Model
                    </Button>
                  </div>
                </div>
              </Card>
            </motion.div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default ModelSelection;
