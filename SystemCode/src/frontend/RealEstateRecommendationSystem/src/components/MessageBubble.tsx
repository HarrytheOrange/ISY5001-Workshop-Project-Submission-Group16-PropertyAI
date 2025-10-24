import { useEffect, useState } from "react";
import { Avatar, AvatarFallback } from "./ui/avatar";

interface MessageBubbleProps {
  message: any;
  isUser: boolean;
  timestamp: string;
}

export function MessageBubble({ message, isUser, timestamp }: MessageBubbleProps) {
  const displayText = typeof message === "object" ? message.content : message;

  const hasImage = displayText?.includes("[[Image]]");
  const cleanedText = displayText?.replace("[[Image]]", "").trim();

  const [imageBlobUrl, setImageBlobUrl] = useState<string | null>(null);

  useEffect(() => {
    if (hasImage) {
      const imageUrl = `http://localhost:8000/temp/static_map_with_legend.png?t=${Date.now()}`;

      // ✅ 强制从后端取图并转为 blob
      fetch(imageUrl, { cache: "no-store" })
        .then((res) => res.blob())
        .then((blob) => {
          const localUrl = URL.createObjectURL(blob);
          setImageBlobUrl(localUrl);
          console.log("🖼️ Loaded unique image:", localUrl);
        })
        .catch((err) => console.error("❌ Failed to load image:", err));
    }
  }, [hasImage, displayText]);

  return (
    <div className={`flex gap-3 mb-4 ${isUser ? "flex-row-reverse" : "flex-row"}`}>
      <Avatar className="w-8 h-8 flex-shrink-0">
        {isUser ? (
          <AvatarFallback className="bg-primary text-primary-foreground">U</AvatarFallback>
        ) : (
          <AvatarFallback className="bg-blue-500 text-white">AI</AvatarFallback>
        )}
      </Avatar>

      <div className={`flex flex-col max-w-[80%] ${isUser ? "items-end" : "items-start"}`}>
        <div
          className={`px-4 py-2 rounded-2xl ${
            isUser
              ? "bg-primary text-primary-foreground rounded-br-sm"
              : "bg-secondary text-secondary-foreground rounded-bl-sm"
          }`}
        >
          <p className="whitespace-pre-wrap">{cleanedText}</p>

          {/* ✅ 每次都显示唯一 blob 图 */}
          {hasImage && imageBlobUrl && (
            <div className="mt-3">
              <img
                src={imageBlobUrl}
                alt="Generated map"
                className="rounded-xl border border-gray-300 shadow-sm max-w-full"
              />
            </div>
          )}
        </div>

        <span className="text-xs text-muted-foreground mt-1 px-2">{timestamp}</span>
      </div>
    </div>
  );
}
