import { useState } from "react";
import { Button } from "./ui/button";
import { Card } from "./ui/card";
import { MessageSquare, MapPin, Palette, Image, Home, ShoppingBag, ChevronUp, ChevronDown } from "lucide-react";
import { ImageWithFallback } from './figma/ImageWithFallback';

interface SuggestedQuestionsProps {
  onQuestionClick: (question: string) => void;
  disabled?: boolean;
}

const textQuestions = [
  {
    icon: Home,
    text: "Compare the Modern Family Home in Oakville vs the Executive Townhouse in Markham",
    category: "comparison"
  },
  {
    icon: ShoppingBag,
    text: "What shopping malls and food courts are near the Downtown Toronto condo?",
    category: "amenities"
  },
  {
    icon: MapPin,
    text: "What are the best restaurants and cafes near Mississauga properties?",
    category: "amenities"
  },
  {
    icon: MessageSquare,
    text: "What's the difference between condo living vs house living?",
    category: "comparison"
  }
];

const visualQuestions = [
  {
    text: "Help me plan the decoration for a modern living room",
    image: "https://images.unsplash.com/photo-1656122381069-9ec666d95cf1?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHxpbnRlcmlvciUyMGRlc2lnbiUyMGxpdmluZyUyMHJvb218ZW58MXx8fHwxNzU3ODYwMTA5fDA&ixlib=rb-4.1.0&q=80&w=1080&utm_source=figma&utm_medium=referral",
    category: "decoration"
  },
  {
    text: "Show me modern kitchen design ideas",
    image: "https://images.unsplash.com/photo-1682888813795-192fca4a10d9?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHxtb2Rlcm4lMjBraXRjaGVuJTIwZGVzaWdufGVufDF8fHx8MTc1Nzg2NDcyNXww&ixlib=rb-4.1.0&q=80&w=1080&utm_source=figma&utm_medium=referral",
    category: "inspiration"
  },
  {
    text: "Show me pictures of bedroom interior designs",
    image: "https://images.unsplash.com/photo-1720247520862-7e4b14176fa8?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHxtb2Rlcm4lMjBsaXZpbmclMjByb29tJTIwaW50ZXJpb3J8ZW58MXx8fHwxNzU3ODM4NDk5fDA&ixlib=rb-4.1.0&q=80&w=1080&utm_source=figma&utm_medium=referral",
    category: "inspiration"
  },
  {
    text: "Compare houses and help me choose the best one",
    image: "https://images.unsplash.com/photo-1690475779509-f83676e56372?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHxob3VzZSUyMGNvbXBhcmlzb24lMjByZWFsJTIwZXN0YXRlfGVufDF8fHx8MTc1NzkwNjEyNnww&ixlib=rb-4.1.0&q=80&w=1080&utm_source=figma&utm_medium=referral",
    category: "comparison"
  }
];

export function SuggestedQuestions({ onQuestionClick, disabled = false }: SuggestedQuestionsProps) {
  const [isCollapsed, setIsCollapsed] = useState(false);
  
  // Get 2 text questions and 2 visual questions randomly
  const getRandomQuestions = () => {
    const shuffledText = [...textQuestions].sort(() => 0.5 - Math.random());
    const shuffledVisual = [...visualQuestions].sort(() => 0.5 - Math.random());
    
    return {
      textQuestions: shuffledText.slice(0, 2),
      visualQuestions: shuffledVisual.slice(0, 2)
    };
  };

  const { textQuestions: displayTextQuestions, visualQuestions: displayVisualQuestions } = getRandomQuestions();

  return (
    <Card className="p-4 m-4 bg-muted/30">
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <MessageSquare className="w-4 h-4 text-muted-foreground" />
          <span className="text-sm font-medium text-muted-foreground">Try asking:</span>
        </div>
        
        <Button
          variant="ghost"
          size="sm"
          onClick={() => setIsCollapsed(!isCollapsed)}
          className="h-8 w-8 p-0 hover:bg-muted"
          disabled={disabled}
        >
          {isCollapsed ? (
            <ChevronUp className="w-4 h-4 text-muted-foreground" />
          ) : (
            <ChevronDown className="w-4 h-4 text-muted-foreground" />
          )}
        </Button>
      </div>
      
      {!isCollapsed && (
        <div className="space-y-3">
          {/* Text-based questions */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
            {displayTextQuestions.map((question, index) => {
              const IconComponent = question.icon;
              return (
                <Button
                  key={`text-${index}`}
                  variant="ghost"
                  className="h-auto p-3 text-left justify-start whitespace-normal text-wrap"
                  onClick={() => onQuestionClick(question.text)}
                  disabled={disabled}
                >
                  <IconComponent className="w-4 h-4 mr-2 flex-shrink-0 mt-0.5" />
                  <span className="text-sm">{question.text}</span>
                </Button>
              );
            })}
          </div>

          {/* Visual preview questions */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            {displayVisualQuestions.map((question, index) => (
              <Button
                key={`visual-${index}`}
                variant="ghost"
                className="h-auto p-0 text-left overflow-hidden"
                onClick={() => onQuestionClick(question.text)}
                disabled={disabled}
              >
                <div className="flex items-center w-full">
                  <div className="relative w-16 h-16 flex-shrink-0">
                    <ImageWithFallback
                      src={question.image}
                      alt={question.text}
                      className="w-full h-full object-cover rounded-l-md"
                    />
                    <div className="absolute inset-0 bg-black/10 rounded-l-md"></div>
                  </div>
                  <div className="flex-1 p-3">
                    <span className="text-sm">{question.text}</span>
                  </div>
                </div>
              </Button>
            ))}
          </div>
        </div>
      )}
    </Card>
  );
}